import time
from typing import Any, cast

import mlflow
import mlflow.sklearn
import numpy as np
from generate_dataset import generate_dataset_with_anomalies
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

mlflow.set_tracking_uri("http://mlflow:5000")
try:
    mlflow.set_experiment("cubestocker_experiment")
except Exception as e:
    print(f"[WARNING] could not set MLflow experiment: {e}")


def nested_cross_validate(X: np.ndarray, y: np.ndarray) -> tuple[list[dict], dict]:
    """
    Performs a 5-fold outer cross-validation using XGBoost Classifier
    model for multi-class anomaly classification.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Multi-class target array of shape (n_samples,), where 0 indicates normal instances and 1+ indicates different anomaly types.

    Returns:
        tuple[list[dict], dict] containing a list of dictionaries, each containing results from one fold and a dictionary corresponding to the best fold, selected by highest AUROC.
    """
    y = np.asarray(y, dtype=int)

    print("[INFO] Starting Outer Cross Validation with 5 folds...")
    print(f"[INFO] Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    skf_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    total_folds = 5
    for fold_idx, (train_i, test_i) in enumerate(skf_outer.split(X, y), start=1):
        print(f"\n[INFO] ========== FOLD {fold_idx}/{total_folds} ==========")
        X_tr, y_tr = X[train_i], y[train_i]
        X_te, y_te = X[test_i], y[test_i]

        # XGBoost parameters for multi-class classification
        xgb_params = {
            "objective": "multi:softprob",  # Multi-class classification with probability output
            "num_class": len(np.unique(y_tr)),  # Number of classes
            "eval_metric": "auc",  # Multi-class auc
            "n_estimators": 100,
            "random_state": 42,
            "verbosity": 0,  # Reduce verbosity
        }

        sample_weights = compute_sample_weight(class_weight="balanced", y=y_tr)

        # Create XGBoost classifier
        xgb_model = XGBClassifier(**xgb_params)

        start_time = time.time()
        model = xgb_model.fit(X_tr, y_tr, sample_weight=sample_weights)
        fit_time = time.time() - start_time

        start_time = time.time()
        y_proba = model.predict_proba(X_te)
        score_time = time.time() - start_time

        # Calculate AUROC for multi-class
        try:
            auroc = roc_auc_score(y_te, y_proba, multi_class="ovr", average="macro")
        except ValueError as e:
            print(f"[WARNING] Could not calculate AUROC for fold {fold_idx}: {e}")
            auroc = 0.0

        fold_results.append({
            "fold": fold_idx,
            "auroc": auroc,
            "fit_time": fit_time,
            "score_time": score_time,
            "model": model,
            "params": xgb_params,
        })

    # choose best fold by (AUROC, then AUPRC)
    best = max(fold_results, key=lambda r: cast(float, r["auroc"]))
    return fold_results, best


class NoRunsFoundError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("No runs found in experiment to select best from.")


class RegisteredModelVersionNotFoundError(RuntimeError):
    def __init__(self, run_id: str):
        super().__init__(f"Could not find a registered model version for run_id {run_id}.")


def _raise_no_runs_error() -> None:
    raise NoRunsFoundError()


def _raise_version_not_found_error(run_id: str) -> None:
    raise RegisteredModelVersionNotFoundError(run_id)


def register_best_model(model_name: str, experiment_id: str, best: dict[str, Any]) -> None:
    client = MlflowClient()

    # Find the single best run across all runs in this experiment
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.cv_mean_auroc DESC"],
        max_results=1,
    )

    if not runs:
        print("[INFO] No runs found matching the criteria.")
        return
    else:
        best_run = runs[0]
        best_run_id = best_run.info.run_id
        best_auroc = best_run.data.metrics.get("cv_mean_auroc")
        print(f"[INFO] Best run: {best_run_id} (cv_mean_auroc={best_auroc})")

        # Determine which registered-model versions correspond to best_run_id
        versions = client.search_model_versions(f"name='{model_name}'")
        best_version_num = None
        for v in versions:
            if v.run_id == best_run_id:
                best_version_num = v.version
                break

        if best_version_num is None:
            _raise_version_not_found_error(best_run_id)

        # Archive previous Production Model, then assign Production to best_version_num.
        for v in versions:
            if v.current_stage == "Production":
                client.transition_model_version_stage(name=model_name, version=v.version, stage="Archived")
                break

        # Transition the best version to "Production"
        client.transition_model_version_stage(name=model_name, version=best_version_num, stage="Production")
        print(f"[INFO] Transitioned version {best_version_num} to 'Production' stage")


def train_and_log_model(noise_pct: float, noise_std: float, noise_type: str, model_name: str) -> str:
    """
    Loads "normal" Cubestocker data.
    Injects synthetic noise → anomalies.
    Runs a 5-fold cross-validation of XGBoost Classifier and logs per-fold metrics + aggregated metrics to MLflow.
    Registers the best-trained XGBoost Classifier model inside a pipeline that includes scaler + XGBoost Classifier.
    Finds that best run across all runs (by cv_mean_auroc), then assigns alias "production" to the corresponding model version (new API).

    Args:
        noise_pct: Percentage of rows to add noise to
        noise_std: Standard deviation of the noise distribution
        noise_type: Type of noise to add (gaussian, laplace, poisson, etc.)
        model_name: Name of the model.
    Returns:
        Run_id of the current training run.
    """
    # Generate dataset with anomalies
    df, feature_cols, anomaly_type_mapping = generate_dataset_with_anomalies(noise_pct, noise_std, noise_type)

    # scale df with scaler
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Split X/y
    X_array = df[feature_cols].to_numpy()
    y_array = df["Anomaly_Type_Label"].to_numpy()

    # MLflow run
    with mlflow.start_run() as run:
        run_id: str = run.info.run_id
        experiment_id = run.info.experiment_id

        mlflow.log_dict(anomaly_type_mapping, "anomaly_type_mapping.json")

        # Cross-validate & log metrics
        folds, best = nested_cross_validate(X_array, y_array)

        # log noise parameters
        mlflow.log_param("noise_pct", noise_pct)
        mlflow.log_param("noise_std", noise_std)
        mlflow.log_param("noise_type", noise_type)
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("num_classes", len(np.unique(y_array)))

        # log metrics per fold
        aurocs, fit_times, score_times = [], [], []
        for f in folds:
            mlflow.log_metric(f"fold{f['fold']}_AUC", f["auroc"])
            mlflow.log_metric(f"fold{f['fold']}_fit_time", f["fit_time"])
            mlflow.log_metric(f"fold{f['fold']}_score_time", f["score_time"])
            aurocs.append(f["auroc"])
            fit_times.append(f["fit_time"])
            score_times.append(f["score_time"])

        # log aggregated metrics
        mlflow.log_metric("cv_mean_auroc", np.mean(aurocs))
        mlflow.log_metric("cv_std_auroc", np.std(aurocs))
        mlflow.log_metric("cv_mean_fit_time", np.mean(fit_times))
        mlflow.log_metric("cv_mean_score_time", np.mean(score_times))

        # log best fold
        mlflow.log_param("best_fold", best["fold"])
        for k, v in best["params"].items():
            mlflow.log_param(f"best_param_{k}", v)

        # Log & register model
        frozen_xgb = best["model"]
        pipeline = Pipeline([
            ("scaler", scaler),
            ("model", frozen_xgb),
        ])
        sample_df = df[feature_cols].iloc[:5]
        sample_preds = pipeline.predict(sample_df)
        signature = infer_signature(sample_df, sample_preds)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=model_name,
            signature=signature,
            input_example=sample_df.to_dict(orient="records"),
        )
        print(f"Model trained and logged. Run ID: {run_id}")
        try:
            register_best_model(model_name, experiment_id, best)
        except Exception as e:
            print(f"[WARNING] Failed to transition model to 'Production': {e}")

        return run_id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and register the model.")
    parser.add_argument("--noise-pct", type=float, default=0.05, help="Percentage of rows to add noise to")
    parser.add_argument("--noise-std", type=float, default=1.5, help="Standard deviation of noise")
    parser.add_argument("--noise-type", type=str, default="laplace", help="Type of noise to add")
    parser.add_argument("--model-name", type=str, default="xgb_multiclass", help="Name of the model")

    args = parser.parse_args()

    run_id = train_and_log_model(args.noise_pct, args.noise_std, args.noise_type, args.model_name)
    print(f"Training complete. MLflow Run ID: {run_id}")
