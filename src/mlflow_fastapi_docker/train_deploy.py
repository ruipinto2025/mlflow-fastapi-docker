import mlflow
import mlflow.sklearn
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("simple_linear_regression_experiment")

MODEL_NAME = "simple_linear_regression_model"


def train_and_log_model():
    """
    Trains a simple linear regression model on synthetic data,
    logs it to the MLflow tracking server, and returns the run_id.
    """
    X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        lr = LinearRegression()
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_test)
        mse = float(np.mean((y_test - y_pred) ** 2))
        r2 = float(lr.score(X_test, y_test))

        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(sk_model=lr, artifact_path="model", registered_model_name=MODEL_NAME)

        print(f"Model trained and logged. Run ID: {run_id}")
        return run_id


# FastAPI app
app = FastAPI(title="Simple Linear Regression API")


class PredictRequest(BaseModel):
    x: float


class PredictResponse(BaseModel):
    prediction: float


# Tenta carregar o modelo ao iniciar
def load_model_from_registry(model_name: str):
    model_uri = f"models:/{model_name}/latest"
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        raise RuntimeError from e
    else:
        return model


try:
    model = load_model_from_registry(MODEL_NAME)
except RuntimeError as e:
    print(f"[WARNING] {e}")
    model = None


@app.get("/")
def read_root():
    return {"message": "Welcome to the Simple Linear Regression API. Send POST to /predict"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        model = load_model_from_registry(MODEL_NAME)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail="Could not load model") from e

    input_array = np.array([[req.x]])
    try:
        pred = model.predict(input_array)
        prediction = float(pred[0])
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

    if args.train:
        run_id = train_and_log_model()
        print(f"Training complete. MLflow Run ID: {run_id}")
    elif args.serve:
        uvicorn.run("mlflow_fastapi_docker.train_deploy:app", host="0.0.0.0", port=8080)  # noqa: S104
    else:
        print("Use --train to train or --serve to serve the app.")
