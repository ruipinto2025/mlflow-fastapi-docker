import time
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.models.signature import infer_signature
from pydantic import BaseModel
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

mlflow.set_tracking_uri("http://mlflow:5000")
try:
    mlflow.set_experiment("cubestocker_experiment")
except Exception as e:
    print(f"[WARNING] could not set MLflow experiment: {e}")

model_name = "lof"


def load_and_preprocess_data(filename: str) -> tuple[pd.DataFrame, StandardScaler, list[str]]:
    """
    Loads Cubestocker normal data, removes unnecessary features, initializes a new column 'Anomaly_Label' as 0 (normal data) and scales the data.

    Args:
        filename: Name of the Cubestocker File to be loaded

    Returns:
        Preprocessed Dataframe, the scaler, and a list of features used for training.
    """
    df = pd.read_csv(f"data/{filename}.csv", delimiter=";", low_memory=False)
    df = df.drop(columns=["new_stamp", "device_id", "date"], errors="ignore")
    df["Anomaly_Label"] = 0
    # convert object columns to float
    obj_cols = df.select_dtypes(include=["object"]).columns
    df[obj_cols] = df[obj_cols].apply(lambda x: pd.to_numeric(x.str.replace(",", ".", regex=True), errors="coerce"))

    feature_cols = [c for c in df.columns if c != "Anomaly_Label"]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, scaler, feature_cols


class UnsupportedNoiseTypeError(ValueError):
    def __init__(self, noise_type: str):
        super().__init__(f"Noise type '{noise_type}' is not supported.")


def add_noise(
    df: pd.DataFrame, noise_pct: float, noise_std: float, noise_type: str, feature_cols: list[str]
) -> pd.DataFrame:
    """
    Add synthetic noise to data to create anomalies.

    Args:
        df: Input DataFrame
        noise_pct: Percentage of rows to add noise to
        noise_std: Standard deviation of the noise distribution
        noise_type: Type of noise to add (gaussian, laplace, poisson, etc.)
        feature_cols: Feature columns to be added noise

    Returns:
        A new DataFrame where `noise_pct * len(df)` randomly chosen rows have had their feature values perturbed according to `noise_type` & `noise_std`. For those rows 'Anomaly_Label' is set to 1.
    """
    # add seed for reproducibility
    rng = np.random.RandomState(0)

    df_noisy = df.copy()
    num_rows = len(df_noisy)
    num_noise = int(num_rows * noise_pct)
    noise_idx = rng.choice(df_noisy.index, size=num_noise, replace=False)

    # -- classical noise types --
    # gaussian, laplace, poisson, gamma, bernoulli
    noise_mean = 0

    if noise_type == "gaussian":
        noise = rng.normal(noise_mean, noise_std, size=(num_noise, len(feature_cols)))
    elif noise_type == "laplace":
        noise = rng.laplace(noise_mean, noise_std, size=(num_noise, len(feature_cols)))
    elif noise_type == "poisson":
        noise = rng.poisson(noise_std, size=(num_noise, len(feature_cols))) - noise_std
    elif noise_type == "gamma":
        shape = 1
        scale = noise_std / np.sqrt(shape)
        noise = rng.gamma(shape, scale, size=(num_noise, len(feature_cols)))
        # Center the gamma noise
        noise = noise - noise.mean()
        if noise.std() != 0:
            noise *= noise_std / noise.std()
    elif noise_type == "bernoulli":
        # Flip sign of ~50% of the features in those rows
        mask = rng.binomial(1, 0.5, size=(num_noise, len(feature_cols))).astype(bool)
        for idx, row_mask in zip(noise_idx, mask):
            for j, col in enumerate(feature_cols):
                if row_mask[j]:
                    df_noisy.loc[idx, col] = -df_noisy.loc[idx, col]
        df_noisy.loc[noise_idx, "Anomaly_Label"] = 1
        return df_noisy
    else:
        raise UnsupportedNoiseTypeError(noise_type)

    # Add the continuous noise for the non-bernoulli types:
    df_noisy.loc[noise_idx, feature_cols] += noise
    df_noisy.loc[noise_idx, "Anomaly_Label"] = 1
    return df_noisy


def nested_cross_validate(X: np.ndarray, y: np.ndarray) -> tuple[list[dict], dict]:
    """
    Performs a 5-fold outer cross-validation using the Local Outlier Factor (LOF)
    model for anomaly detection.

    In each fold, the model is trained only on the normal class (where y == 0). LOF is used with fixed parameters (no hyperparameter optimization). Performance is evaluated using AUROC and AUPRC on the test fold.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Binary target array of shape (n_samples,), where 0 indicates normal instances and 1 indicates anomalies.

    Returns:
        tuple[list[dict], dict] containing a list of dictionaries, each containing results from one fold and a dictionary corresponding to the best fold, selected by highest AUROC (with AUPRC as tie-breaker).
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

        # without optimization for now
        best_params: dict[str, Any] = {}

        lof = LocalOutlierFactor(novelty=True, n_neighbors=20)

        # training only with normal data
        mask = y_tr == 0
        start_time = time.time()
        model = lof.fit(X_tr[mask])
        fit_time = time.time() - start_time

        start_time = time.time()
        raw_scores = model.decision_function(X_te)
        score_time = time.time() - start_time
        scores = (1 - raw_scores) / 2.0

        auroc = roc_auc_score(y_te, scores)
        auprc = average_precision_score(y_te, scores)

        fold_results.append({
            "fold": fold_idx,
            "auroc": auroc,
            "auprc": auprc,
            "fit_time": fit_time,
            "score_time": score_time,
            "model": model,
            "params": best_params,
        })

    # choose best fold by (AUROC, then AUPRC)
    best = max(fold_results, key=lambda r: (r["auroc"], r["auprc"]))
    return fold_results, best


def train_and_log_model(noise_pct: float, noise_std: float, noise_type: str, model_name: str) -> str:
    """
    Loads “normal” Cubestocker data.
    Injects synthetic noise → anomalies.
    Runs a 5-fold cross-validation of LOF (novelty-mode) and logs per-fold metrics + aggregated metrics to MLflow.
    Registers the best-trained LOF model inside a pipeline that includes ColumnTransformer (selector + scaler) → LOF.
    Returns the MLflow run_id.

    Args:
        noise_pct: Percentage of rows to add noise to
        noise_std: Standard deviation of the noise distribution
        noise_type: Type of noise to add (gaussian, laplace, poisson, etc.)
        model_name: Name of the model.
    Returns:
        The Mlflow run id.
    """

    df, scaler, feature_cols = load_and_preprocess_data("TRACKING_20250328_a_20250403")
    df = add_noise(df, noise_pct, noise_std, noise_type, feature_cols)

    X = df[feature_cols].values
    y = np.asarray(df["Anomaly_Label"].tolist(), dtype=int)

    with mlflow.start_run() as run:
        run_id: str = run.info.run_id

        folds, best = nested_cross_validate(X, y)

        # log metrics per fold
        aurocs, auprcs, fit_times, score_times = [], [], [], []
        for f in folds:
            mlflow.log_metric(f"fold{f['fold']}_AUC", f["auroc"])
            mlflow.log_metric(f"fold{f['fold']}_AUPRC", f["auprc"])
            mlflow.log_metric(f"fold{f['fold']}_fit_time", f["fit_time"])
            mlflow.log_metric(f"fold{f['fold']}_score_time", f["score_time"])
            aurocs.append(f["auroc"])
            auprcs.append(f["auprc"])
            fit_times.append(f["fit_time"])
            score_times.append(f["score_time"])

        # log aggregated metrics
        mlflow.log_metric("cv_mean_auroc", np.mean(aurocs))
        mlflow.log_metric("cv_std_auroc", np.std(aurocs))
        mlflow.log_metric("cv_mean_auprc", np.mean(auprcs))
        mlflow.log_metric("cv_std_auprc", np.std(auprcs))
        mlflow.log_metric("cv_mean_fit_time", np.mean(fit_times))
        mlflow.log_metric("cv_mean_score_time", np.mean(score_times))

        # log best fold
        mlflow.log_param("best_fold", best["fold"])
        for k, v in best["params"].items():
            mlflow.log_param(f"best_param_{k}", v)

        frozen_lof = best["model"]
        pipeline = Pipeline([
            ("scaler", scaler),
            ("model", frozen_lof),
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
        return run_id


# FastAPI app
app = FastAPI(title="LocalOutlierFactor API")


class PredictRequest(BaseModel):
    """Request model with all 50 feature columns (53 original - 3 dropped columns)"""

    # CR1 Motor X measurements
    cr1_mtx_temperature: float
    cr1_mtx_current: float
    cr1_mtx_power: float
    cr1_mtx_speed: float
    cr1_mtx_torque: float
    cr1_mtx_position: float

    # CR1 Motor Y measurements
    cr1_mty_temperature: float
    cr1_mty_current: float
    cr1_mty_power: float
    cr1_mty_speed: float
    cr1_mty_torque: float
    cr1_mty_position: float

    # CR1 Motor Z measurements
    cr1_mtz_temperature: float
    cr1_mtz_current: float
    cr1_mtz_power: float
    cr1_mtz_speed: float
    cr1_mtz_torque: float
    cr1_mtz_position: float

    # CR1 Belt and Load measurements
    cr1_y_beltvibration_xangle: float
    cr1_y_beltvibration_yangle: float
    cr1_y_load: float

    # CR2 Motor X measurements
    cr2_mtx_temperature: float
    cr2_mtx_current: float
    cr2_mtx_power: float
    cr2_mtx_speed: float
    cr2_mtx_torque: float
    cr2_mtx_position: float

    # CR2 Motor Y measurements
    cr2_mty_temperature: float
    cr2_mty_current: float
    cr2_mty_power: float
    cr2_mty_speed: float
    cr2_mty_torque: float
    cr2_mty_position: float

    # CR2 Motor Z measurements
    cr2_mtz_temperature: float
    cr2_mtz_current: float
    cr2_mtz_power: float
    cr2_mtz_speed: float
    cr2_mtz_torque: float
    cr2_mtz_position: float

    # CR2 Belt and Load measurements
    cr2_y_beltvibration_xangle: float
    cr2_y_beltvibration_yangle: float
    cr2_y_load: float

    # Mode and Status columns
    cr1_mode: float
    cr2_mode: float
    cr1_has_request: float
    cr2_has_request: float
    cr1_has_carrier: float
    cr2_has_carrier: float
    cr1_to_exit_full: float
    cr2_to_exit_full: float


class PredictResponse(BaseModel):
    """Response model for anomaly detection: 0 = normal, 1 = anomaly"""

    prediction: int  # 0 for normal, 1 for anomaly


def load_model_from_registry(model_name: str) -> Any:
    """
    Loads the latest version of `model_name` from the MLflow model registry.
    Raises RuntimeError if loading fails.

    Args:
        model_name: Name of the OCC model
    Returns:
        The latest fitted model.
    """
    model_uri = f"models:/{model_name}/latest"
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        raise RuntimeError from e
    return model


try:
    model = load_model_from_registry(model_name)
except RuntimeError as e:
    print(f"[WARNING] {e}")
    model = None


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "Welcome to the Local Outlier Factor API. Send POST to /predict"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        model = load_model_from_registry(model_name)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail="Could not load model") from e

    # Convert request to DataFrame with correct column order
    input_data = {
        "cr1_mtx_temperature": req.cr1_mtx_temperature,
        "cr1_mtx_current": req.cr1_mtx_current,
        "cr1_mtx_power": req.cr1_mtx_power,
        "cr1_mtx_speed": req.cr1_mtx_speed,
        "cr1_mtx_torque": req.cr1_mtx_torque,
        "cr1_mtx_position": req.cr1_mtx_position,
        "cr1_mty_temperature": req.cr1_mty_temperature,
        "cr1_mty_current": req.cr1_mty_current,
        "cr1_mty_power": req.cr1_mty_power,
        "cr1_mty_speed": req.cr1_mty_speed,
        "cr1_mty_torque": req.cr1_mty_torque,
        "cr1_mty_position": req.cr1_mty_position,
        "cr1_mtz_temperature": req.cr1_mtz_temperature,
        "cr1_mtz_current": req.cr1_mtz_current,
        "cr1_mtz_power": req.cr1_mtz_power,
        "cr1_mtz_speed": req.cr1_mtz_speed,
        "cr1_mtz_torque": req.cr1_mtz_torque,
        "cr1_mtz_position": req.cr1_mtz_position,
        "cr1_y_beltvibration_xangle": req.cr1_y_beltvibration_xangle,
        "cr1_y_beltvibration_yangle": req.cr1_y_beltvibration_yangle,
        "cr1_y_load": req.cr1_y_load,
        "cr2_mtx_temperature": req.cr2_mtx_temperature,
        "cr2_mtx_current": req.cr2_mtx_current,
        "cr2_mtx_power": req.cr2_mtx_power,
        "cr2_mtx_speed": req.cr2_mtx_speed,
        "cr2_mtx_torque": req.cr2_mtx_torque,
        "cr2_mtx_position": req.cr2_mtx_position,
        "cr2_mty_temperature": req.cr2_mty_temperature,
        "cr2_mty_current": req.cr2_mty_current,
        "cr2_mty_power": req.cr2_mty_power,
        "cr2_mty_speed": req.cr2_mty_speed,
        "cr2_mty_torque": req.cr2_mty_torque,
        "cr2_mty_position": req.cr2_mty_position,
        "cr2_mtz_temperature": req.cr2_mtz_temperature,
        "cr2_mtz_current": req.cr2_mtz_current,
        "cr2_mtz_power": req.cr2_mtz_power,
        "cr2_mtz_speed": req.cr2_mtz_speed,
        "cr2_mtz_torque": req.cr2_mtz_torque,
        "cr2_mtz_position": req.cr2_mtz_position,
        "cr2_y_beltvibration_xangle": req.cr2_y_beltvibration_xangle,
        "cr2_y_beltvibration_yangle": req.cr2_y_beltvibration_yangle,
        "cr2_y_load": req.cr2_y_load,
        "cr1_mode": req.cr1_mode,
        "cr2_mode": req.cr2_mode,
        "cr1_has_request": req.cr1_has_request,
        "cr2_has_request": req.cr2_has_request,
        "cr1_has_carrier": req.cr1_has_carrier,
        "cr2_has_carrier": req.cr2_has_carrier,
        "cr1_to_exit_full": req.cr1_to_exit_full,
        "cr2_to_exit_full": req.cr2_to_exit_full,
    }

    input_df = pd.DataFrame([input_data])

    try:
        # Get prediction (-1 for anomaly, 1 for normal)
        pred = model.predict(input_df)
        # Convert to 0/1 format: -1 (anomaly) -> 1, 1 (normal) -> 0
        prediction = 1 if pred[0] == -1 else 0
        return PredictResponse(prediction=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Prediction error") from e


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Train a linear model or run the API.")
    parser.add_argument("--train", action="store_true", help="Train and register the model.")
    parser.add_argument("--serve", action="store_true", help="Serve the FastAPI app.")
    args = parser.parse_args()

    noise_pct = 0.05
    noise_std = 1.5
    noise_type = "laplace"
    if args.train:
        run_id = train_and_log_model(noise_pct, noise_std, noise_type, model_name)
        print(f"Training complete. MLflow Run ID: {run_id}")
    elif args.serve:
        uvicorn.run("mlflow_fastapi_docker.train_deploy:app", host="0.0.0.0", port=8080)  # noqa: S104
    else:
        print("Use --train to train or --serve to serve the app.")
