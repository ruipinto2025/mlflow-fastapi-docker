from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, Optional

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from generate_dataset import (
    ANOMALY_DETAILS_MAP,
    decision_layer,
    get_anomaly_labels,
)
from mlflow.tracking import MlflowClient
from pydantic import BaseModel

mlflow.set_tracking_uri("http://mlflow:5000")

model_name = "xgb_multiclass"
model = None


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


class RegisteredModelVersionNotFoundError(RuntimeError):
    def __init__(self, run_id: str):
        super().__init__(f"Could not find a registered model version for run_id {run_id}.")


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
def read_root() -> dict[str, Any]:
    return {
        "message": "Welcome to the XGBoost Multi-class Anomaly Detection API.",
        "endpoints": {
            "predict": "POST /predict - Send sensor data for anomaly detection",
            "health": "GET /health - Check API health status",
        },
    }


@app.get("/health")
def health_check() -> dict[str, Any]:
    """Health check endpoint"""
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available")
    return {"status": "healthy", "model_loaded": True}


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
    import uvicorn

    uvicorn.run("mlflow_fastapi_docker.deploy:app", host="0.0.0.0", port=8080, reload=True, log_level="info")  # noqa: S104
