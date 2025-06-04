# Install uv
FROM python:3.10-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Change the working directory to the `app` directory
WORKDIR /app

# Copy the lockfile and `pyproject.toml` into the image
COPY uv.lock /app/uv.lock
COPY pyproject.toml /app/pyproject.toml

# Install dependencies
RUN uv sync --locked --no-install-project --no-dev

# Copy the project into the image
COPY . /app

# Sync the project
RUN uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH"

#CMD [ "python", "src/mlflow_fastapi_docker/train_deploy.py" ]
CMD [ "python", "src/mlflow_fastapi_docker/train_deploy.py", "--serve"]
