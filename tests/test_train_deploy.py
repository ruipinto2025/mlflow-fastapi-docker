# tests/test_train_deploy.py
import mlflow
import numpy as np
import pytest
from fastapi.testclient import TestClient

from mlflow_fastapi_docker.train_deploy import app, train_and_log_model

mlflow.set_tracking_uri = lambda uri: None
mlflow.set_experiment = lambda name: None


class DummyModel:
    """
    A minimal stand-in for an sklearn type model.
    Its .predict() always returns [x * 2] so we can assert on it.
    """

    def predict(self, arr: np.ndarray) -> np.ndarray:
        # assume input is [[x]]; return array([x * 2])
        return np.array([arr[0][0] * 2])


@pytest.fixture(autouse=True)
def stub_out_load_model(monkeypatch):
    """
    By default, if load_model_from_registry is called, make it raise RuntimeError
    (as if no model is registered). Individual tests can override this behavior.
    """
    monkeypatch.setattr("mlflow.sklearn.load_model", lambda uri: (_ for _ in ()).throw(RuntimeError("no model")))
    yield


def test_read_root_returns_welcome_message():
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"message": "Welcome to the Simple Linear Regression API. Send POST to /predict"}


class ModelNotFoundError(Exception):
    def __init__(self, model_name: str) -> None:
        msg = f"Model '{model_name}' not found"
        super().__init__(msg)


def test_predict_when_no_model_registered(monkeypatch):
    """
    If load_model_from_registry raises RuntimeError, the endpoint should return 503.
    """

    def fake_load_fail(name: str):
        raise ModelNotFoundError(name)

    monkeypatch.setattr("mlflow_fastapi_docker.train_deploy.load_model_from_registry", fake_load_fail)

    client = TestClient(app)
    payload = {"x": 3.14}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 503
    assert resp.json() == {"detail": "Could not load model"}


def test_predict_success_with_dummy_model(monkeypatch):
    """
    Monkey patch load_model_from_registry to return DummyModel.
    POST /predict with {"x": 2.5} should return prediction = 5.0.
    """
    monkeypatch.setattr("mlflow_fastapi_docker.train_deploy.load_model_from_registry", lambda name: DummyModel())

    client = TestClient(app)
    payload = {"x": 2.5}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200

    body = resp.json()
    assert "prediction" in body
    # DummyModel.predict([[2.5]]) → array([5.0])
    assert pytest.approx(body["prediction"], rel=1e-6) == 5.0


def test_train_and_log_model_returns_run_id(monkeypatch):
    """
    Patch mlflow.start_run so train_and_log_model() returns a fake run_id.
    """
    import mlflow as real_mlflow

    class DummyRunInfo:
        def __init__(self, run_id):
            self.run_id = run_id

    class DummyRun:
        def __init__(self):
            self.info = DummyRunInfo("FAKE_RUN_123")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

    # 1) Patch mlflow.start_run() to return DummyRun whose .info.run_id == "FAKE_RUN_123"
    monkeypatch.setattr(real_mlflow, "start_run", lambda: DummyRun())

    # 2) Patch mlflow.log_metric and mlflow.sklearn.log_model to no-ops:
    monkeypatch.setattr(real_mlflow, "log_metric", lambda key, val: None)
    monkeypatch.setattr(real_mlflow.sklearn, "log_model", lambda sk_model, artifact_path, registered_model_name: None)

    run_id = train_and_log_model()
    assert run_id == "FAKE_RUN_123"
