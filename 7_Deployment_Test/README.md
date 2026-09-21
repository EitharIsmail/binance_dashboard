# Binance Direction Classifier — Deployment & Integration Test

This folder contains the **containerized deployment environment** for the Binance Direction Classifier.

It brings together three services:

* **MLflow** — model registry and model-serving artifact source
* **FastAPI** — online prediction API
* **Streamlit** — user-facing prediction dashboard

The services are orchestrated using **Docker Compose** and communicate through an internal Docker network.

The purpose of this layer is to verify that the trained model can move from the MLflow registry into the online API and ultimately be consumed through a simple user interface.

---

## 1. Deployment Architecture

The complete deployment consists of:

```text id="6s7c9a"
                         Browser
                            │
                            │ http://localhost:8501
                            ▼
                 ┌─────────────────────┐
                 │     Streamlit       │
                 │     Dashboard       │
                 │                     │
                 │  "Predict" button   │
                 └──────────┬──────────┘
                            │
                            │ http://api:8000
                            ▼
                 ┌─────────────────────┐
                 │      FastAPI        │
                 │    Prediction API   │
                 │                     │
                 │  /health            │
                 │  /predict           │
                 └──────────┬──────────┘
                            │
                            │ MLflow
                            ▼
                 ┌─────────────────────┐
                 │       MLflow        │
                 │  Model Registry     │
                 │                     │
                 │    Production       │
                 │      Model          │
                 └─────────────────────┘
```

The API also communicates directly with Binance to obtain recent market data:

```text id="tx7c4a"
                       Binance
                          │
                          │ Recent market data
                          ▼
                    FastAPI API
                          │
                          ▼
                  Feature Engineering
                          │
                          ▼
                  Production Model
                          │
                          ▼
                     Prediction
                          │
                          ▼
                    Streamlit UI
```

---

# 2. What This Deployment Environment Solves

The training pipeline and online API can work independently, but an actual application needs all components to communicate correctly.

This deployment environment tests the complete chain:

```text id="3f5g8e"
Trained Model
     ↓
MLflow Registry
     ↓
FastAPI
     ↓
Streamlit
     ↓
User
```

It verifies that:

* MLflow starts correctly,
* the registered model is accessible,
* FastAPI can load the Production model,
* FastAPI can retrieve recent Binance data,
* feature engineering works inside the container,
* the model can generate a prediction,
* Streamlit can communicate with FastAPI,
* all services can run together using Docker Compose.

---

# 3. Services

The `docker-compose.yml` defines three services.

| Service     | Technology | Port | Responsibility                      |
| ----------- | ---------- | ---: | ----------------------------------- |
| `mlflow`    | MLflow     | 5000 | Model registry and artifact serving |
| `api`       | FastAPI    | 8000 | Online inference                    |
| `dashboard` | Streamlit  | 8501 | User interface                      |

The dependency chain is:

```text id="y0x8f3"
MLflow
  ↓
API
  ↓
Dashboard
```

Docker Compose starts the services according to these dependencies.

---

# 4. MLflow Service

The MLflow service uses the official MLflow container image:

```text id="w8t7e1"
ghcr.io/mlflow/mlflow:latest
```

It starts an MLflow tracking server on:

```text id="6yq4zv"
http://localhost:5000
```

Inside the Docker network, other services access it using:

```text id="w0x8z5"
http://mlflow:5000
```

The important distinction is:

```text id="7e3n5q"
localhost:5000
```

is used from the host machine, while:

```text id="d7m0q2"
mlflow:5000
```

is used by containers communicating with the MLflow service.

---

## 4.1 MLflow Storage

The deployment environment stores the MLflow database in:

```text id="s1v5k4"
7_Deployment_Test/mlflow/mlflow.db
```

The directory is mounted into the container:

```text id="4z6b2c"
./mlflow:/mlflow
```

The MLflow backend store is therefore:

```text id="t2m9x1"
sqlite:////mlflow/mlflow.db
```

The deployment environment also mounts:

```text id="6g1x0r"
./mlruns:/mlruns
```

for MLflow artifacts.

This allows the model registry and artifacts to persist outside the container.

---

# 5. Why a Separate Deployment MLflow Exists

The project has an MLflow environment associated with the training pipeline and another one used by this deployment environment.

Conceptually:

```text id="f4s1z7"
3_Pipeline
     │
     ▼
Training MLflow
     │
     │ selected model
     ▼
Deployment MLflow
     │
     ▼
Production
     │
     ▼
FastAPI
```

The deployment environment therefore acts as a controlled runtime environment rather than directly depending on the local MLflow database used during development/training.

The model artifact present under `mlruns/` represents the model available to this deployment environment.

---

# 6. FastAPI Service

The API is built from:

```text id="b9k6p3"
../4_Deploy_Online/api/Dockerfile
```

The Docker Compose configuration uses:

```yaml id="z0m3v8"
build:
  context: ../4_Deploy_Online
  dockerfile: api/Dockerfile
```

This means the deployment test does not duplicate the API source code.

Instead:

```text id="n5c2r9"
7_Deployment_Test
       │
       └── docker-compose.yml
                    │
                    ▼
             4_Deploy_Online
                    │
                    ▼
                FastAPI
```

This is useful because the integration environment tests the same API implementation that belongs to the online deployment layer.

---

# 7. API → MLflow Communication

The API receives this environment variable:

```text id="6w3p8d"
MLFLOW_TRACKING_URI=http://mlflow:5000
```

Therefore, inside the Docker network:

```text id="a1f6q0"
FastAPI
   │
   │ http://mlflow:5000
   ▼
MLflow
```

At application startup, FastAPI loads the Production models from MLflow.

For example:

```text id="m8v4s2"
models:/BTCUSDT_30m_classifier/Production
```

The API then keeps the loaded model in memory for inference.

---

# 8. FastAPI → Binance Communication

The API obtains recent market data directly from Binance.

For a prediction request, the flow is:

```text id="q8j4t1"
Streamlit
    ↓
FastAPI
    ↓
Binance Vision + Binance REST API
    ↓
Recent BTCUSDT candles
    ↓
Feature engineering
    ↓
MLflow Production model
    ↓
Prediction
```

The deployment implementation combines:

* the latest available Binance Vision monthly archive,
* Binance's live REST kline endpoint.

Live data takes precedence when timestamps overlap.

This allows the API to obtain recent market information while maintaining compatibility with the historical data format used during training.

---

# 9. Streamlit Dashboard

The user-facing application is located in:

```text id="v1q7r3"
dashboard/
├── dashboard.py
├── Dockerfile
└── requirements.txt
```

The dashboard is intentionally simple in the current MVP.

It currently supports:

```text id="s4n9x2"
Asset: BTCUSDT
Horizon: 30 minutes
```

There are no user controls for selecting another asset or horizon yet.

The main interaction is:

```text id="r2w7c5"
▶ Predict Bitcoin's next 30 minutes
```

---

# 10. Dashboard → API Communication

The dashboard receives:

```text id="0v6n2k"
ONLINE_API=http://api:8000
```

This is an important Docker networking detail.

The Streamlit Python process runs **inside its own container**.

Therefore it should not use:

```text id="h4b1z9"
http://localhost:8000
```

to communicate with the API container.

Instead, it uses the Docker Compose service name:

```text id="d6q3m8"
http://api:8000
```

The communication is:

```text id="n0p5x7"
Streamlit container
       │
       │ http://api:8000
       ▼
FastAPI container
```

The user accesses Streamlit from the host through:

```text id="u7k2c1"
http://localhost:8501
```

So there are two different networking perspectives:

```text id="b4m9r2"
Browser → localhost:8501 → Streamlit

Streamlit → api:8000 → FastAPI

FastAPI → mlflow:5000 → MLflow
```

This distinction is important when debugging Docker networking.

---

# 11. Dashboard Health Check

Before allowing a prediction, the dashboard checks:

```text id="k3x8p0"
GET /health
```

The API reports which Production horizons are currently loaded.

The dashboard then checks whether the required `30m` model is available.

Conceptually:

```text id="z5d1q8"
Dashboard
    │
    ▼
GET /health
    │
    ├── API unavailable
    │       ↓
    │   Show error
    │
    ├── API available
    │       │
    │       └── 30m model unavailable
    │                    ↓
    │                Disable prediction
    │
    └── 30m model available
                 ↓
             Enable button
```

This prevents the user interface from attempting a prediction when the required Production model is unavailable.

---

# 12. Prediction Request

When the user clicks the prediction button, Streamlit sends:

```json id="k8x1m5"
{
  "horizon": "30m"
}
```

to:

```text id="z6v3r1"
POST http://api:8000/predict
```

The API then:

1. selects the `30m` Production model,
2. fetches recent BTCUSDT data,
3. computes the required features,
4. generates the prediction,
5. converts the class into a human-readable direction,
6. returns the model name and version.

Example API response:

```json id="j9c2w6"
{
  "prediction": "Bull",
  "model_name": "BTCUSDT_30m_classifier",
  "model_version": "1"
}
```

The dashboard then displays the result to the user.

---

# 13. End-to-End Prediction Flow

The complete request can be represented as:

```text id="r5m1q8"
User
 │
 │ Click "Predict"
 ▼
Streamlit
 │
 │ POST /predict
 ▼
FastAPI
 │
 ├── Select 30m Production model
 │
 ├── Fetch recent Binance data
 │
 ├── Compute technical features
 │
 ├── Run preprocessing
 │
 ├── Run classifier
 │
 └── Convert class → direction
 │
 ▼
JSON response
 │
 ▼
Streamlit
 │
 ▼
Bull / Neutral / Bear
```

---

# 14. Docker Compose Networking

Docker Compose creates a shared internal network for the services.

The services can therefore communicate using their service names:

```text id="d9s4m2"
mlflow:5000
api:8000
```

while the host accesses them through published ports:

```text id="q3w8n6"
localhost:5000
localhost:8000
localhost:8501
```

The mapping is:

| Service   | Container address | Host address     |
| --------- | ----------------- | ---------------- |
| MLflow    | `mlflow:5000`     | `localhost:5000` |
| FastAPI   | `api:8000`        | `localhost:8000` |
| Streamlit | `dashboard:8501`  | `localhost:8501` |

The container address should be used for **service-to-service communication**.

The host address should be used from the **browser or host machine**.

---

# 15. Health Checks

Docker Compose defines health checks for the API and dashboard.

### API

The container checks:

```text id="w3f7n1"
http://localhost:8000/health
```

every 30 seconds.

### Dashboard

The container checks:

```text id="a6q2v8"
http://localhost:8501/_stcore/health
```

every 30 seconds.

These checks allow Docker to detect whether the services are responding.

---

# 16. Starting the Deployment

From this directory:

```bash id="n7x3m9"
cd 7_Deployment_Test
```

start all services:

```bash id="r1v6k4"
docker compose up --build
```

The `--build` option ensures that the API and dashboard images are rebuilt from the current source.

To run in the background:

```bash id="p5q8w2"
docker compose up --build -d
```

---

# 17. Checking Running Containers

Run:

```bash id="s8d2k5"
docker compose ps
```

You should see the three services:

```text
mlflow
api
dashboard
```

To inspect logs:

```bash id="c7m1v4"
docker compose logs
```

Or inspect an individual service:

```bash id="x9q3b6"
docker compose logs api
```

```bash id="h2v7n1"
docker compose logs dashboard
```

```bash id="k5w8d3"
docker compose logs mlflow
```

Follow logs continuously with:

```bash id="j4m9p2"
docker compose logs -f api
```

---

# 18. Accessing the Services

Once the containers are running:

### Streamlit Dashboard

```text id="q6r2x8"
http://localhost:8501
```

This is the main user interface.

### FastAPI

```text id="m3v7k1"
http://localhost:8000
```

### Swagger API Documentation

```text id="t8n4c5"
http://localhost:8000/docs
```

### FastAPI Health

```text id="y2p6s9"
http://localhost:8000/health
```

### MLflow

```text id="f7m1q3"
http://localhost:5000
```

---

# 19. Testing the Deployment

A useful testing sequence is:

### Test 1 — Check containers

```bash id="u5c8r2"
docker compose ps
```

Confirm that the three services are running.

### Test 2 — Check API health

Open:

```text id="b1n6z4"
http://localhost:8000/health
```

The response should show the loaded Production models.

For the current MVP, the expected deployed horizon is:

```text id="p4w7m2"
30m
```

### Test 3 — Test Swagger

Open:

```text id="k8q3v6"
http://localhost:8000/docs
```

Use:

```text
POST /predict
```

with:

```json id="a2j9s5"
{
  "horizon": "30m"
}
```

### Test 4 — Test the dashboard

Open:

```text id="z6r1x4"
http://localhost:8501
```

The dashboard should:

1. report the API/model status,
2. enable the prediction button,
3. request a prediction,
4. display Bull, Neutral, or Bear,
5. display the model name and version.

---

# 20. Stopping the Deployment

Stop the services with:

```bash id="v3m8q1"
docker compose down
```

This stops and removes the containers while leaving the mounted MLflow database and artifacts on the host.

To rebuild everything after code changes:

```bash id="c1x7n5"
docker compose down
docker compose up --build
```

---

# 21. Dashboard Design

The current dashboard deliberately has a small interface.

It displays:

### System status

Whether the API and required Production model are available.

### Prediction

One of:

```text
Bull
Neutral
Bear
```

### Prediction timestamp

The UTC time when the request was made.

### Forecast time

The target time 30 minutes after the prediction request.

### Model information

The registered MLflow model and version used for the prediction.

This provides basic prediction traceability.

---

# 22. Current Deployment Configuration

The current MVP is intentionally fixed to:

| Configuration           | Value                 |
| ----------------------- | --------------------- |
| Asset                   | Bitcoin / BTCUSDT     |
| Candle interval         | 15 minutes            |
| Prediction horizon      | 30 minutes            |
| Prediction classes      | Bull / Neutral / Bear |
| API                     | FastAPI               |
| Dashboard               | Streamlit             |
| Model registry          | MLflow                |
| Container orchestration | Docker Compose        |
| Dashboard port          | 8501                  |
| API port                | 8000                  |
| MLflow port             | 5000                  |

The API itself has support for multiple horizons, but the current Streamlit dashboard intentionally exposes only the `30m` Bitcoin prediction workflow.

---

# 23. Environment Variables

### API

The Compose environment provides:

```text id="e5w2n9"
MLFLOW_TRACKING_URI=http://mlflow:5000
ROOT_PATH=/api
```

`MLFLOW_TRACKING_URI` tells the API where the deployment MLflow server is located.

---

### Dashboard

The dashboard receives:

```text id="m7q4x1"
ONLINE_API=http://api:8000
```

This tells Streamlit where the FastAPI service is located inside the Docker network.

The dashboard also receives Streamlit server configuration for:

* headless operation,
* port,
* bind address,
* CORS,
* XSRF protection.

---

# 24. Persistence

The Compose configuration mounts two host directories:

```text id="h8p2r6"
./mlflow:/mlflow
./mlruns:/mlruns
```

This means MLflow's database and artifacts are not stored only inside the temporary container filesystem.

The deployment structure is:

```text id="d2n7k4"
7_Deployment_Test/
│
├── mlflow/
│   └── mlflow.db
│
└── mlruns/
    └── ...
```

Therefore, recreating the MLflow container does not automatically remove these host-side files.

---

# 25. Deployment Artifact

The deployment environment currently contains a model artifact under:

```text id="q9m3v7"
mlruns/
└── 1/
    └── models/
        └── ...
            └── artifacts/
                ├── MLmodel
                ├── model.skops
                ├── conda.yaml
                ├── python_env.yaml
                └── requirements.txt
```

The important artifact is the serialized ML model:

```text id="w4c8n2"
model.skops
```

The accompanying environment files describe the dependencies and model configuration associated with that artifact.

The API ultimately loads the model through MLflow rather than manually opening `model.skops`.

---

# 26. Integration Boundaries

Each component has a clearly defined responsibility:

```text id="s6x1m8"
┌─────────────────┐
│    Streamlit    │
│                 │
│ Presentation    │
└────────┬────────┘
         │ HTTP
         ▼
┌─────────────────┐
│     FastAPI     │
│                 │
│ Online Inference│
└────────┬────────┘
         │ MLflow
         ▼
┌─────────────────┐
│     MLflow      │
│                 │
│ Model Registry  │
└─────────────────┘
```

This separation makes it possible to change one layer without redesigning the others.

For example:

* the dashboard can be redesigned without changing the classifier,
* the API can be updated without changing the MLflow model artifact,
* the model can be replaced through the registry without rebuilding the dashboard.

---

# 27. Relationship to the Other Project Components

The broader project is divided into several stages.

```text id="m1v5q8"
3_Pipeline
    │
    │ Train + evaluate
    ▼
Training MLflow
    │
    │ Model deployment/migration
    ▼
7_Deployment_Test
    │
    ├── MLflow
    │
    ├── FastAPI
    │
    └── Streamlit
    │
    ▼
User-facing prediction
```

The responsibilities are therefore:

| Component                    | Responsibility                            |
| ---------------------------- | ----------------------------------------- |
| `3_Pipeline`                 | Train and evaluate models                 |
| `4_Deploy_Online`            | Implement the FastAPI serving layer       |
| `7_Deployment_Test`          | Run the complete containerized deployment |
| `8_CI_CD`                    | Automate build/test/deployment workflows  |
| `9_Monitoring_Observability` | Monitor the deployed system               |

---

# 28. Troubleshooting

## API cannot load a model

Check:

```bash id="r8m2c5"
docker compose logs api
```

Look for errors related to:

* MLflow connectivity,
* model name,
* model version,
* Production stage,
* missing artifacts.

Then check:

```text
http://localhost:5000
```

to verify that the deployment MLflow server is running.

---

## Dashboard cannot reach API

Check:

```bash id="x3n7k1"
docker compose logs dashboard
```

Inside Docker, the dashboard should use:

```text id="b5q9m4"
http://api:8000
```

not:

```text id="z7c2p6"
http://localhost:8000
```

---

## API is running but prediction returns 503

Check:

```text id="f1m8r3"
http://localhost:8000/health
```

If the `30m` horizon is not present in `loaded_horizons`, the API does not currently have a Production model available for that horizon.

---

## Containers are running but the dashboard shows an error

Check the dependency chain:

```text id="v6q2n9"
MLflow
  ↓
API
  ↓
Dashboard
```

Then inspect each service:

```bash id="c4r8m1"
docker compose logs mlflow
docker compose logs api
docker compose logs dashboard
```

---

# 29. Current MVP Scope

The current deployment demonstrates a complete end-to-end ML inference workflow:

```text id="n9x3k7"
Historical Training
       ↓
Model Registry
       ↓
Dockerized MLflow
       ↓
Dockerized FastAPI
       ↓
Recent Binance Data
       ↓
Feature Engineering
       ↓
Production Model
       ↓
Prediction
       ↓
Streamlit Dashboard
```

The current user-facing workflow is intentionally limited to:

```text id="k5m1r8"
BTCUSDT
   +
30-minute horizon
   +
Bull / Neutral / Bear
```

This provides a small, testable deployment before expanding the application.

---

# 30. Future Improvements

Possible improvements to this deployment layer include:

* user-selectable prediction horizons,
* support for multiple cryptocurrencies,
* prediction history,
* confidence/probability display,
* API authentication,
* request rate limiting,
* structured API logging,
* centralized monitoring,
* model-performance monitoring,
* data-quality monitoring,
* automated model updates,
* CI/CD deployment,
* production-grade secrets management,
* HTTPS/reverse proxy,
* persistent prediction storage,
* richer dashboard visualizations.

For production use, additional infrastructure would also be required around security, reliability, observability, and operational controls.

---

# 31. Important Note

The dashboard presents the output of a machine-learning classification model.

A result such as:

```text id="x2n7c4"
Bull
```

is a model prediction based on the latest available market features. It does not guarantee a future price movement or trading profit.

The application should therefore be understood as an **ML-based decision-support prototype** rather than a guaranteed trading system.

---

## Quick Reference

### Start

```bash id="q7m2v9"
cd 7_Deployment_Test
docker compose up --build
```

### Run in background

```bash id="w4c8n1"
docker compose up --build -d
```

### Check services

```bash id="m9x3r6"
docker compose ps
```

### View logs

```bash id="p2k7v5"
docker compose logs -f
```

### Open dashboard

```text id="h6n1q8"
http://localhost:8501
```

### Open API

```text id="d3r8m2"
http://localhost:8000
```

### Open Swagger

```text id="v5q1x7"
http://localhost:8000/docs
```

### Open MLflow

```text id="c8m4n9"
http://localhost:5000
```

### Stop

```bash id="j2w6p3"
docker compose down
```

---

## Summary

`7_Deployment_Test` is the integration environment that connects the project's trained ML model to a complete online application:

```text id="s3m8q1"
                ┌──────────────┐
                │    MLflow    │
                │    Registry  │
                └──────┬───────┘
                       │
                       ▼
                ┌──────────────┐
                │   FastAPI    │
                │  /predict    │
                └──────┬───────┘
                       │
                       ▼
                ┌──────────────┐
                │  Streamlit   │
                │  Dashboard   │
                └──────┬───────┘
                       │
                       ▼
                     User
```

Docker Compose packages these components into a reproducible environment where model serving, API inference, and the user interface can be tested together.
