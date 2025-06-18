import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, Optional, cast

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from pydantic import BaseModel
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

model_name = "xgb_multiclass"
model = None


def load_and_preprocess_data(filenames: list[str]) -> tuple[pd.DataFrame, StandardScaler, list[str]]:
    """
    Loads Cubestocker normal data, removes unnecessary features, initializes a new column 'Anomaly_Label' as 0 (normal data) and scales the data.

    Args:
        filenames: Names of the Cubestocker files to be loaded

    Returns:
        Preprocessed Dataframe, the scaler, and a list of features used for training.
    """
    dfs = []
    for fname in filenames:
        dfi = pd.read_parquet(f"data/{fname}.parquet", engine="fastparquet")
        dfi = dfi.drop(columns=["new_stamp", "device_id", "date"], errors="ignore")
        dfi["Anomaly_Label"] = 0
        dfs.append(dfi)

    df = pd.concat(dfs, ignore_index=True)

    feature_cols = [c for c in df.columns if c != "Anomaly_Label"]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, scaler, feature_cols


class UnsupportedNoiseTypeError(ValueError):
    def __init__(self, noise_type: str):
        super().__init__(f"Noise type '{noise_type}' is not supported.")


def _get_noise_columns(feature_cols: list[str], exclude_cols: list[str]) -> list[str]:
    """Filter out excluded columns from feature columns."""
    return [col for col in feature_cols if col not in exclude_cols]


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
    num_rows = len(df)
    num_noise = int(num_rows * noise_pct)
    noise_idx = rng.choice(df.index, size=num_noise, replace=False)

    # -- classical noise types --
    # gaussian, laplace, poisson, gamma, bernoulli
    noise_mean = 0

    exclude_cols = [
        "cr1_mode",
        "cr2_mode",
        "cr1_has_request",
        "cr2_has_request",
        "cr1_has_carrier",
        "cr2_has_carrier",
        "cr1_to_exit_full",
        "cr2_to_exit_full",
    ]

    noise_columns = _get_noise_columns(feature_cols, exclude_cols)

    if noise_type == "gaussian":
        noise = rng.normal(noise_mean, noise_std, size=(num_noise, len(noise_columns)))
    elif noise_type == "laplace":
        noise = rng.laplace(noise_mean, noise_std, size=(num_noise, len(noise_columns)))
    elif noise_type == "poisson":
        noise = rng.poisson(noise_std, size=(num_noise, len(noise_columns))) - noise_std
    elif noise_type == "gamma":
        shape = 1
        scale = noise_std / np.sqrt(shape)
        noise = rng.gamma(shape, scale, size=(num_noise, len(noise_columns)))
        # Center the gamma noise
        noise = noise - noise.mean()
        if noise.std() != 0:
            noise *= noise_std / noise.std()
    elif noise_type == "bernoulli":
        # Flip sign of ~50% of the features in those rows
        mask = rng.binomial(1, 0.5, size=(num_noise, len(noise_columns))).astype(bool)
        for idx, row_mask in zip(noise_idx, mask):
            for j, col in enumerate(noise_columns):
                if row_mask[j]:
                    df.loc[idx, col] = -df.loc[idx, col]
        df.loc[noise_idx, "Anomaly_Label"] = 1
        return df
    else:
        raise UnsupportedNoiseTypeError(noise_type)

    # Add the continuous noise for the non-bernoulli types:
    df.loc[noise_idx, noise_columns] += noise
    df.loc[noise_idx, "Anomaly_Label"] = 1
    return df


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


# Temperature limits (Float, ºC)
TEMP_LIMITS: dict[str, tuple[float, float]] = {"x": (20.00, 25.00), "y": (20.00, 70.00), "z": (20.00, 22.00)}

# Current limits (Float, A)
CURRENT_LIMITS: dict[str, tuple[float, float]] = {"x": (0.00, 20.00), "y": (0.00, 10.00), "z": (0.00, 5.00)}

# Speed limits (Float, RPM)
SPEED_LIMITS: dict[str, tuple[float, float]] = {"x": (-2000, 2000.00), "y": (-3000, 3000.00), "z": (-1500, 1500.00)}

# Position limits (Integer, mm)
POSITION_LIMITS: dict[str, tuple[int, int]] = {"x": (2148, 41755), "y": (-902, 432400), "z": (-70901, 70901)}

# Torque limits (Float, Nm)
TORQUE_LIMITS: dict[str, tuple[float, float]] = {"x": (-60.00, 55.00), "y": (-15.00, 15.00), "z": (-5.00, 5.00)}

# Power limits (Float, kWh)
POWER_LIMITS: dict[str, tuple[float, float]] = {"x": (-5.5, 5.5), "y": (-2.85, 2.85), "z": (-0.4, 0.4)}

# Belt vibration angle limits (Float, °)
ANGLE_X_LIMITS: tuple[float, float] = (174.00, 186.00)
ANGLE_Y_LIMITS: tuple[float, float] = (74.00, 106.00)


def check_motor_overheating(row: dict[str, Any], prefix: str = "cr2_") -> Optional[str]:
    """
    Motor Overheating:
    If the temperature exceeds its maximum in any axis, flag a Motor Overheating anomaly.
    """
    for axis in ["x", "y", "z"]:
        temp = row[f"{prefix}mt{axis}_temperature"]
        _, temp_max = TEMP_LIMITS[axis]
        if temp > temp_max:
            return f"Motor Overheating ({prefix}mtx/mty/mtz)"
    return None


def check_motor_overcurrent(row: dict[str, Any], prefix: str = "cr2_") -> Optional[str]:
    """
    Motor Overcurrent
    If the current exceeds its maximum in any axis, flag a Motor Overcurrent anomaly.
    """
    for axis in ["x", "y", "z"]:
        current = row[f"{prefix}mt{axis}_current"]
        _, curr_max = CURRENT_LIMITS[axis]
        if current > curr_max:
            return f"Motor Overcurrent ({prefix}mtx/mty/mtz)"
    return None


def check_anomalous_motor_position(row: dict[str, Any], prefix: str = "cr2_") -> Optional[str]:
    """
    Anomalous Motor Position:
    If the motor position is out of its allowed range, flag the anomaly "anomalous motor position".
    """
    for axis in ["x", "y", "z"]:
        pos = row[f"{prefix}mt{axis}_position"]
        pos_min, pos_max = POSITION_LIMITS[axis]

        if pos < pos_min or pos > pos_max:
            return f"Anomalous Motor Position ({prefix}mtx/mty/mtz)"
    return None


def check_high_speed_operation(row: dict[str, Any], prefix: str = "cr2_") -> Optional[str]:
    """
    High Speed Operation:
    If the speed in any axis is outside the allowed range while the current remains within its normal range,
    flag an High Speed Operation anomaly.
    """
    for axis in ["x", "y", "z"]:
        speed = row[f"{prefix}mt{axis}_speed"]
        s_min, s_max = SPEED_LIMITS[axis]
        if speed < s_min or speed > s_max:
            return f"High Speed Operation ({prefix}mtx/mty/mtz)"
    return None


def check_abnormal_motor_torque(row: dict[str, Any], prefix: str = "cr2_") -> Optional[str]:
    """
    Abnormal Motor Torque:
    If the torque in any axis deviates beyond its allowed range flag an Abnormal Motor Torque anomaly.
    """
    for axis in ["x", "y", "z"]:
        torque = row[f"{prefix}mt{axis}_torque"]
        t_min, t_max = TORQUE_LIMITS[axis]
        if torque < t_min or torque > t_max:
            return f"Abnormal Motor Torque ({prefix}mtx/mty/mtz)"
    return None


def check_abnormal_power_variation(row: dict[str, Any], prefix: str = "cr2_") -> Optional[str]:
    """
    Abnormal Power Variation:
    If the power reading in any axis is outside its normal range flag a Abnormal Power Variation anomaly.
    """
    for axis in ["x", "y", "z"]:
        power = row[f"{prefix}mt{axis}_power"]
        p_min, p_max = POWER_LIMITS[axis]
        if power < p_min or power > p_max:
            return f"Abnormal Power Variation ({prefix}mtx/mty/mtz)"
    return None


def check_high_vibration(row: dict[str, Any], prefix: str = "cr2_") -> Optional[str]:
    """
    High Vibration:
    If the vibration angles (from belt sensors) are outside their permitted ranges flag a High Vibration anomaly.
    """

    xangle = row[f"{prefix}y_beltvibration_xangle"]
    yangle = row[f"{prefix}y_beltvibration_yangle"]
    excessive_vibration = (
        xangle < ANGLE_X_LIMITS[0]
        or xangle > ANGLE_X_LIMITS[1]
        or yangle < ANGLE_Y_LIMITS[0]
        or yangle > ANGLE_Y_LIMITS[1]
    )

    if excessive_vibration:
        return f"High Vibration ({prefix}y_beltvibration)"
    return None


ANOMALY_DETAILS_MAP: dict[str, dict[str, Any]] = {
    "Motor Overcurrent": {
        "severity_label": "Critical",
        "action": "Stop immediately to prevent damage and collisions. Reduce operation intensity and analyze whether system configuration adjustments are needed",
        "explanation": "Current spikes outside the normal range characterize a critical overload.",
    },
    "Motor Overheating": {
        "severity_label": "Critical",
        "action": "Activate the cooling protocol",
        "explanation": "Temperature has exceeded the normal value, indicating overheating.",
    },
    "Anomalous Motor Position": {
        "severity_label": "Critical",
        "action": "Stop immediately to prevent damage and collisions. Reduce operation intensity and analyze whether system configuration adjustments are needed",
        "explanation": "The motor position is outside the normal range, possibly due to excessive speed.",
    },
    "High Speed Operation": {
        "severity_label": "Critical",
        "action": "Stop immediately to prevent damage and collisions. Reduce operation intensity and analyze whether system configuration adjustments are needed",
        "explanation": "Speed exceeds the normal limit, suggesting a possible malfunction or control failure.",
    },
    "Abnormal Motor Torque": {
        "severity_label": "Critical",
        "action": "Stop immediately to prevent damage and collisions. Reduce operation intensity and analyze whether system configuration adjustments are needed",
        "explanation": "The force applied to the motor is outside the normal range.",
    },
    "High Vibration": {
        "severity_label": "High",
        "action": "Perform a mechanical inspection of the crane; check for wear and angular alignment",
        "explanation": "Vibration angles deviate significantly from normal values, indicating high vibration.",
    },
    "Abnormal Power Variation": {
        "severity_label": "Medium",
        "action": "Stop immediately to prevent damage and collisions. Reduce operation intensity and analyze whether system configuration adjustments are needed",
        "explanation": "Motor power is outside its normal values.",
    },
}


def decision_layer(anomaly_labels: list[str]) -> list[dict[str, Any]]:
    """
    Prioritize anomalies based on the following order:
      1. Motor Overcurrent (Critical)
      2. Motor Overheating (Critical)
      3. Anomalous Motor Position (Critical)
      4. High Speed Operation (Critical)
      5. Abnormal Motor Torque (Critical)
      6. High Vibration (High)
      7. Abnormal Power Variation (Medium)

    Returns a list with a single dictionary containing details for the highest priority anomaly.
    """

    priority_order = [
        "Motor Overcurrent",
        "Motor Overheating",
        "Anomalous Motor Position",
        "High Speed Operation",
        "Abnormal Motor Torque",
        "High Vibration",
        "Abnormal Power Variation",
    ]

    for anomaly_type in priority_order:
        matching = [label for label in anomaly_labels if anomaly_type in label]
        if matching:
            info = ANOMALY_DETAILS_MAP[anomaly_type]
            return [
                {
                    "anomaly": matching[0],
                    "severity": info["severity_label"],
                    "action": info["action"],
                    "explanation": info["explanation"],
                }
            ]
    return []


def get_anomaly_labels(row: pd.Series) -> list[str]:
    """
    Apply domain-specific checks for both Crane 1 (prefix 'cr1_') and Crane 2 (prefix 'cr2_'),
    and return a list of anomaly labels detected.
    """
    labels = []
    data = row.to_dict()
    for prefix in ["cr1_", "cr2_"]:
        for check in (
            check_motor_overcurrent,
            check_motor_overheating,
            check_anomalous_motor_position,
            check_high_speed_operation,
            check_abnormal_power_variation,
            check_abnormal_motor_torque,
            check_high_vibration,
        ):
            msg = check(data, prefix=prefix)
            if msg is not None:
                labels.append(msg)
    return labels


def get_numerical_anomaly_label(row: pd.Series, mapping: dict[str, int]) -> int:
    symbolic_labels = row["symbolic_labels"]

    if not symbolic_labels:
        return mapping["Normal"]
    else:
        # If symbolic labels are found, apply the priority logic
        priority_order = [
            "Motor Overcurrent",
            "Motor Overheating",
            "Anomalous Motor Position",
            "High Speed Operation",
            "Abnormal Motor Torque",
            "High Vibration",
            "Abnormal Power Variation",
        ]
        for anomaly_type in priority_order:
            matching_labels = [label for label in symbolic_labels if anomaly_type in label]
            if matching_labels:
                return mapping[anomaly_type]
        # Fallback in case a symbolic label was generated but didn't match priority_order (shouldn't happen)
        return mapping["Normal"]


class AnomalyDetails(BaseModel):
    """Detailed information about detected anomaly"""

    severity: str
    action: str
    explanation: str


def get_anomaly_details_from_data(row_data: dict) -> Optional[AnomalyDetails]:
    """
    Apply domain-specific rules to get detailed anomaly information.

    Args:
        row_data: dictionary containing the sensor data for one sample

    Returns:
        AnomalyDetails if anomaly detected, None otherwise
    """
    # Convert dict to pandas Series for compatibility with existing functions
    row = pd.Series(row_data)

    # Get anomaly labels using existing domain logic
    anomaly_labels = get_anomaly_labels(row)

    if not anomaly_labels:
        return None

    # Use existing decision layer to get prioritized anomaly details
    decisions = decision_layer(anomaly_labels)

    if decisions:
        decision = decisions[0]
        return AnomalyDetails(
            severity=decision["severity"],
            action=decision["action"],
            explanation=decision["explanation"],
        )

    return None


# Global variable to cache the mapping
_anomaly_mapping_cache = None


def load_anomaly_type_mapping() -> dict[int, str]:
    """
    Load the anomaly type mapping from the latest MLflow run.
    Returns a dictionary mapping numerical labels to anomaly type names.
    """
    try:
        client = MlflowClient()
        experiment = client.get_experiment_by_name("cubestocker_experiment")
        if experiment is None:
            return {0: "Normal"}

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.cv_mean_auroc DESC"],
            max_results=1,
        )

        if not runs:
            return {0: "Normal"}

        best_run = runs[0]
        artifacts = client.list_artifacts(best_run.info.run_id)

        # Look for the anomaly_type_mapping.json artifact
        mapping_artifact = None
        for artifact in artifacts:
            if artifact.path == "anomaly_type_mapping.json":
                mapping_artifact = artifact
                break

        if mapping_artifact:
            import json

            mapping_path = client.download_artifacts(best_run.info.run_id, "anomaly_type_mapping.json")
            with open(mapping_path) as f:
                forward_mapping = json.load(f)  # name -> number

            # Reverse the mapping to get number -> name
            reverse_mapping = {v: k for k, v in forward_mapping.items()}
            return reverse_mapping
        else:
            return {0: "Normal"}

    except Exception as e:
        print(f"[WARNING] Could not load anomaly type mapping: {e}")
        return {0: "Normal"}


def get_anomaly_mapping() -> dict[int, str]:
    """Get cached anomaly mapping or load it"""
    global _anomaly_mapping_cache
    if _anomaly_mapping_cache is None:
        _anomaly_mapping_cache = load_anomaly_type_mapping()
    return _anomaly_mapping_cache


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
        print(f"[INFO] Best run: {best_run_id} (cv_mean_auroc={best_auroc}")

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
    Loads “normal” Cubestocker data.
    Injects synthetic noise → anomalies.
    Runs a 5-fold cross-validation of XGBoost Classifier and logs per-fold metrics + aggregated metrics to MLflow.
    Registers the best-trained XGBoost Classifier model inside a pipeline that includes scaler + XGBoost Classifier.
    Finds that best run across all runs (by cv_mean_auroc), then assigns alias “production” to the corresponding model version (new API).

    Args:
        noise_pct: Percentage of rows to add noise to
        noise_std: Standard deviation of the noise distribution
        noise_type: Type of noise to add (gaussian, laplace, poisson, etc.)
        model_name: Name of the model.
    Returns:
        Run_id of the current training run.
    """
    # 1) Load & add noise
    filenames = [
        "TRACKING_20250502_a_20250508",
        "TRACKING_20250509_a_20250515",
        "TRACKING_20250516_a_20250522",
        "TRACKING_20250523_a_20250529",
    ]
    df, scaler, feature_cols = load_and_preprocess_data(filenames)
    df = add_noise(df, noise_pct, noise_std, noise_type, feature_cols)

    # 2) Reverse-scale & symbolic labeling
    reversed_data = scaler.inverse_transform(df[feature_cols])
    reversed_df = pd.DataFrame(reversed_data, columns=feature_cols)
    reversed_df["Anomaly_Label"] = df["Anomaly_Label"]
    reversed_df["symbolic_labels"] = reversed_df.apply(get_anomaly_labels, axis=1)

    # 3) Build anomaly-type → int mapping
    all_anomaly_types = set()
    for labels_list in reversed_df["symbolic_labels"]:
        for label in labels_list:
            # Extract only the anomaly type string, excluding the parenthesized part
            anomaly_type = label.split(" (")[0]
            all_anomaly_types.add(anomaly_type)

    sorted_anomaly_types = sorted(all_anomaly_types)
    anomaly_type_mapping = {anom_type: i + 1 for i, anom_type in enumerate(sorted_anomaly_types)}
    anomaly_type_mapping["Normal"] = 0  # Add label for normal data

    # 4) Create numeric labels
    reversed_df["Anomaly_Type_Label"] = 0  # Initialize with 0 (Normal)
    reversed_df["Anomaly_Type_Label"] = reversed_df.apply(
        lambda row: get_numerical_anomaly_label(row, anomaly_type_mapping), axis=1
    )
    reversed_df = reversed_df.drop(columns=["symbolic_labels", "Anomaly_Label"])

    # 5) Rescale features and split X/y
    feature_cols_reversed = [c for c in reversed_df.columns if c != "Anomaly_Type_Label"]
    X_before_scaling = reversed_df[feature_cols_reversed]
    y = reversed_df["Anomaly_Type_Label"]

    X_array = scaler.fit_transform(X_before_scaling)
    y_array = y.to_numpy()

    # 6) MLflow run
    with mlflow.start_run() as run:
        run_id: str = run.info.run_id
        experiment_id = run.info.experiment_id

        mlflow.log_dict(anomaly_type_mapping, "anomaly_type_mapping.json")

        # 7) Cross-validate & log metrics
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

        # 8) Log & register model
        frozen_xgb = best["model"]
        pipeline = Pipeline([
            ("scaler", scaler),
            ("model", frozen_xgb),
        ])
        sample_df = df[feature_cols_reversed].iloc[:5]
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


# ------------FastAPI app---------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global model
    # --- Startup: load the model once before handling any requests ---
    try:
        model = load_model_from_registry(model_name)
    except RuntimeError as e:
        # If loading fails, leave model = None so /predict returns 503
        print(f"[WARNING] {e}")
        model = None

    yield  # <- app is now ready to receive requests

    # --- Shutdown: cleanup if needed (e.g., release resources) ---
    model = None


# FastAPI app
app = FastAPI(lifespan=lifespan, title="Welcome to the XGBoost Multi-class Anomaly Detection API")


class PredictRequest(BaseModel):
    """Request model with all 53 original columns"""

    device_id: int

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
    cr1_mode: int
    cr2_mode: int
    cr1_has_request: int
    cr2_has_request: int
    cr1_has_carrier: int
    cr2_has_carrier: int
    cr1_to_exit_full: int
    cr2_to_exit_full: int

    # time
    date: int
    new_stamp: int


class PredictResponse(BaseModel):
    """Response model for multi-class anomaly detection"""

    prediction_label: str  # Human-readable label
    anomaly_details: Optional[AnomalyDetails] = None  # Detailed info if anomaly detected


class ModelLoadError(RuntimeError):
    def __init__(self, version: str, original_exception: Exception):
        message = f"Failed to load model version {version}: {original_exception}"
        super().__init__(message)
        self.version = version
        self.original_exception = original_exception


class NoProductionModelError(RuntimeError):
    def __init__(self, model_name: str):
        message = f"No model version in 'Production' stage for model '{model_name}'"
        super().__init__(message)
        self.model_name = model_name


def load_model_from_registry(model_name: str) -> Any:
    """
    Loads the version of `model_name` that is currently in the 'Production' stage.
    Raises RuntimeError if no version is in 'Production' or if loading fails.

    Args:
        model_name: Name of the OCC model
    Returns:
        The model in the 'Production' stage
    """
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    for v in versions:
        if v.current_stage == "Production":
            model_uri = f"models:/{model_name}/{v.version}"
            try:
                model = mlflow.sklearn.load_model(model_uri)
            except Exception as e:
                raise ModelLoadError(v.version, e) from e
            return model
    raise NoProductionModelError(model_name)


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "Welcome to the XGBoost Multi-class Anomaly Detection API. Send POST to /predict"}


@app.post("/predict", response_model=list[PredictResponse])
def predict(reqs: list[PredictRequest]) -> list[PredictResponse]:
    global model
    if model is None:
        # If startup-time loading failed (or model not registered), return 503
        raise HTTPException(status_code=503, detail="Model is not available")

    records = [r.dict() for r in reqs]
    df = pd.DataFrame(records).drop(columns=["device_id", "date", "new_stamp"])

    feat_order = list(model.named_steps["scaler"].feature_names_in_)
    input_df = df[feat_order]

    try:
        class_predictions = model.predict(input_df)
        anomaly_mapping = get_anomaly_mapping()
        responses = []

        for _i, pred_class in enumerate(class_predictions):
            # Get human-readable label
            pred_label = anomaly_mapping.get(pred_class, f"Class_{pred_class}")

            # Get detailed anomaly information if it's an anomaly (class != 0)
            anomaly_details = None
            if pred_class != 0 and pred_label in ANOMALY_DETAILS_MAP:
                info = ANOMALY_DETAILS_MAP[pred_label]
                anomaly_details = AnomalyDetails(
                    severity=info["severity_label"],
                    action=info["action"],
                    explanation=info["explanation"],
                )

            response = PredictResponse(prediction_label=pred_label, anomaly_details=anomaly_details)
            responses.append(response)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Prediction error") from e
    else:
        return responses


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Train a model or run the API.")
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
