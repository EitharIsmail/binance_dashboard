# Binance Direction Classifier — Deployment & Integration Test

This folder contains the **containerized deployment environment** for the Binance Direction Classifier.

It brings together three services:

* **MLflow** — model registry and model artifact serving
* **FastAPI** — online prediction API
* **Streamlit** — user-facing prediction dashboard

The services are orchestrated using **Docker Compose** and communicate through an internal Docker network.

The purpose of this layer is to verify that the trained model can move from the MLflow registry into the online API and ultimately be consumed through a simple user interface.

---

# 1. Deployment Architecture

The complete deployment consists of:

```text
                         Browser
                            │
                            │ VPS host port 1063
                            ▼
                 ┌─────────────────────┐
                 │     Streamlit       │
                 │     Dashboard       │
                 │     :8501           │
                 │                     │
                 │  "Predict" button   │
                 └──────────┬──────────┘
                            │
                            │ http://api:8000
                            ▼
                 ┌─────────────────────┐
                 │      FastAPI        │
                 │    Prediction API   │
                 │       :8000         │
                 │                     │
                 │  /health            │
                 │  /predict           │
                 └──────────┬──────────┘
                            │
                            │ http://mlflow:1060
                            ▼
                 ┌─────────────────────┐
                 │       MLflow        │
                 │  Model Registry     │
                 │       :1060         │
                 │                     │
                 │    Production       │
                 │      Model          │
                 └─────────────────────┘
```

The API also communicates directly with Binance to obtain recent market data:

```text
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

```text
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

| Service     | Technology | VPS Host Port | Container Port | Responsibility                      |
| ----------- | ---------- | ------------: | -------------: | ----------------------------------- |
| `mlflow`    | MLflow     |      **1060** |       **1060** | Model registry and artifact serving |
| `api`       | FastAPI    |      **1061** |       **8000** | Online inference                    |
| `dashboard` | Streamlit  |      **1063** |       **8501** | User interface                      |

The dependency chain is:

```text
MLflow
  ↓
API
  ↓
Dashboard
```

Docker Compose manages the services and their dependencies. Health checks are used to determine whether services are responding.

---

# 4. Host Ports vs Docker Network Ports

There are two networking perspectives in this deployment.

## From the host/VPS

The services use the assigned VPS host ports:

```text
MLflow     → 1060
FastAPI    → 1061
Dashboard  → 1063
```

If these ports are exposed externally, they can be accessed using:

```text
MLflow     → http://35.202.67.240:1060
FastAPI    → http://35.202.67.240:1061
Dashboard  → http://35.202.67.240:1063
```

## Between containers

Docker services communicate using their **Compose service names and container ports**:

```text
dashboard → http://api:8000
api       → http://mlflow:1060
```

Therefore:

```text
Browser
   │
   │ VPS host port 1063
   ▼
Streamlit :8501
   │
   │ api:8000
   ▼
FastAPI :8000
   │
   │ mlflow:1060
   ▼
MLflow :1060
```

**Important:** `localhost` inside a container refers to that same container, not another Docker service.

The host port and container port do not have to be the same.

For example:

```text
VPS 1061 → API container 8000
VPS 1063 → Dashboard container 8501
```

---

# 5. MLflow Service

The MLflow service uses:

```text
ghcr.io/mlflow/mlflow:latest
```

It starts an MLflow tracking server on container port:

```text
1060
```

The host port is also:

```text
1060
```

When directly exposed, the host accesses MLflow through:

```text
http://35.202.67.240:1060
```

When accessing MLflow locally through the SSH tunnel described later, use:

```text
http://localhost:1060
```

Inside the Docker network, the API accesses MLflow using:

```text
http://mlflow:1060
```

---

## 5.1 MLflow Storage

The deployment environment stores the MLflow database in:

```text
Deployment_test/mlflow/mlflow.db
```

The directory is mounted into the container:

```text
./mlflow:/mlflow
```

The MLflow backend store is:

```text
sqlite:////mlflow/mlflow.db
```

The deployment environment also mounts:

```text
./mlruns:/mlruns
```

for MLflow artifacts.

This allows the model registry database and artifacts to persist outside the container.

---

# 6. Why a Separate Deployment MLflow Exists

The project has an MLflow environment associated with the training pipeline and another one used by this deployment environment.

Conceptually:

```text
Pipeline
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

The deployment environment therefore acts as a controlled runtime environment rather than directly depending on the MLflow database used during development and training.

The model artifact available under `mlruns/` represents the model available to this deployment environment.

---

# 7. FastAPI Service

The API is built from:

```text
../Online/api/Dockerfile
```

The Docker Compose configuration uses:

```yaml
build:
  context: ../Online
  dockerfile: api/Dockerfile
```

This means the deployment test does not duplicate the API source code.

Instead:

```text
Deployment_test
       │
       └── docker-compose.yml
                    │
                    ▼
             Online
                    │
                    ▼
                FastAPI
```

This allows the integration environment to test the same API implementation used by the online deployment layer.

The FastAPI container listens on:

```text
8000
```

The VPS host maps:

```text
1061 → 8000
```

---

# 8. API → MLflow Communication

The API receives:

```text
MLFLOW_TRACKING_URI=http://mlflow:1060
```

Therefore:

```text
FastAPI :8000
   │
   │ http://mlflow:1060
   ▼
MLflow :1060
```

At application startup, FastAPI loads the Production models from MLflow.

For example:

```text
models:/BTCUSDT_30m_classifier/Production
```

The API then keeps the loaded model in memory for inference.

---

# 9. FastAPI → Binance Communication

The API obtains recent market data directly from Binance.

For a prediction request, the flow is:

```text
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

# 10. Streamlit Dashboard

The user-facing application is located in:

```text
dashboard/
├── dashboard.py
├── Dockerfile
└── requirements.txt
```

The dashboard is intentionally simple in the current MVP.

It currently supports:

```text
Asset: BTCUSDT
Horizon: 30 minutes
```

There are no user controls for selecting another asset or horizon yet.

The main interaction is:

```text
▶ Predict Bitcoin's next 30 minutes
```

The Streamlit container listens on:

```text
8501
```

The VPS host maps:

```text
1063 → 8501
```

---

# 11. Dashboard → API Communication

The dashboard receives:

```text
ONLINE_API=http://api:8000
```

The Streamlit Python process runs **inside its own container**.

Therefore it should not use:

```text
http://localhost:1061
```

to communicate with the API container.

Instead, it uses the Docker Compose service name and the API's container port:

```text
http://api:8000
```

The communication is:

```text
Streamlit container :8501
       │
       │ http://api:8000
       ▼
FastAPI container :8000
```

The user accesses Streamlit from the host through:

```text
http://localhost:1063
```

when running locally, or:

```text
http://35.202.67.240:1063
```

when accessing the VPS directly.

---

# 12. Dashboard Health Check

Before allowing a prediction, the dashboard checks:

```text
GET /health
```

The API reports which Production horizons are currently loaded.

The dashboard then checks whether the required `30m` model is available.

Conceptually:

```text
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

# 13. Prediction Request

When the user clicks the prediction button, Streamlit sends:

```json
{
  "horizon": "30m"
}
```

to:

```text
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

```json
{
  "prediction": "Bull",
  "model_name": "BTCUSDT_30m_classifier",
  "model_version": "1"
}
```

The dashboard then displays the result to the user.

---

# 14. End-to-End Prediction Flow

The complete request can be represented as:

```text
User
 │
 │ Click "Predict"
 ▼
Streamlit :8501
 │
 │ POST /predict
 ▼
FastAPI :8000
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

# 15. Docker Compose Networking

Docker Compose creates a shared internal network for the services.

The services communicate using their Compose service names and **container ports**:

```text
mlflow:1060
api:8000
dashboard:8501
```

The host mappings are:

| Service   | Docker address   | VPS host address     |
| --------- | ---------------- | -------------------- |
| MLflow    | `mlflow:1060`    | `35.202.67.240:1060` |
| FastAPI   | `api:8000`       | `35.202.67.240:1061` |
| Streamlit | `dashboard:8501` | `35.202.67.240:1063` |

The Docker address should be used for **service-to-service communication**.

The host address should be used from the **browser or external host** when the corresponding host port is exposed.

---

# 16. Health Checks

Docker Compose defines health checks for the API and dashboard.

### API

The container checks:

```text
http://localhost:8000/health
```

every 30 seconds.

### Dashboard

The container checks:

```text
http://localhost:8501/_stcore/health
```

every 30 seconds.

These checks allow Docker to detect whether the services are responding.

---

# 17. Starting the Deployment on the VPS

The deployment runs on the shared VPS using the assigned host ports:

```text
MLflow:    1060
FastAPI:   1061
Dashboard: 1063
```

Connect to the VPS:

```bash
ssh eithar@35.202.67.240
```

Then:

```bash
cd ~/binance_dashboard/Deployment_test
```

Start the deployment:

```bash
docker compose up --build
```

The `--build` option rebuilds the API and dashboard images from the current source.

To run in the background:

```bash
docker compose up --build -d
```

---

# 18. Checking Running Containers

Run:

```bash
docker compose ps
```

You should see:

```text
mlflow
api
dashboard
```

To inspect all logs:

```bash
docker compose logs
```

Or inspect an individual service:

```bash
docker compose logs api
```

```bash
docker compose logs dashboard
```

```bash
docker compose logs mlflow
```

Follow logs continuously with:

```bash
docker compose logs -f api
```

---

# 19. Accessing the VPS Services

Once the containers are running, and assuming the host ports are externally exposed:

### Streamlit Dashboard

```text
http://35.202.67.240:1063
```

### FastAPI

```text
http://35.202.67.240:1061
```

### Swagger API Documentation

```text
http://35.202.67.240:1061/docs
```

### FastAPI Health

```text
http://35.202.67.240:1061/health
```

### MLflow

```text
http://35.202.67.240:1060
```

> These are the current deployment-test host ports. In the next deployment stage, Nginx can place the dashboard and API behind a domain/reverse proxy and HTTPS instead of exposing these ports directly.

---

# 20. SSH Tunnel for Local MLflow Access

The local training pipeline can connect to the VPS MLflow server through an SSH tunnel.

Run this command on the **local laptop**, not inside the VPS:

```bash
ssh -N -L 1060:localhost:1060 eithar@35.202.67.240
```

This creates:

```text
Local laptop
localhost:1060
      │
      │ SSH tunnel
      ▼
VPS
localhost:1060
      │
      ▼
MLflow
```

Keep the SSH tunnel terminal open while using the connection.

Verify it from another local terminal:

```bash
curl http://localhost:1060
```

You should receive an MLflow response.

---

# 21. Connecting the Local Training Pipeline to VPS MLflow

On the local laptop:

```bash
cd ~/Documents/Coding/binance_dashboard
```

Activate the environment:

```bash
source .venv/bin/activate
```

Set the MLflow tracking URI:

```bash
export MLFLOW_TRACKING_URI=http://localhost:1060
```

Verify:

```bash
echo $MLFLOW_TRACKING_URI
```

Expected:

```text
http://localhost:1060
```

The local pipeline can then communicate with the VPS MLflow server through the SSH tunnel.

Run the pipeline:

```bash
python Pipeline/flow.py
```

The resulting communication is:

```text
Local Training Pipeline
          │
          │ localhost:1060
          ▼
     SSH Tunnel
          │
          ▼
    VPS MLflow:1060
```

---

# 22. Testing the Deployment

A useful testing sequence is:

### Test 1 — Check containers

```bash
docker compose ps
```

Confirm that the three services are running.

### Test 2 — Check MLflow

Open:

```text
http://35.202.67.240:1060
```

or, through the SSH tunnel from the laptop:

```text
http://localhost:1060
```

### Test 3 — Check API health

Open:

```text
http://35.202.67.240:1061/health
```

The response should show the loaded Production models.

For the current MVP, the expected deployed horizon is:

```text
30m
```

### Test 4 — Test Swagger

Open:

```text
http://35.202.67.240:1061/docs
```

Use:

```text
POST /predict
```

with:

```json
{
  "horizon": "30m"
}
```

### Test 5 — Test the dashboard

Open:

```text
http://35.202.67.240:1063
```

The dashboard should:

1. report the API/model status,
2. enable the prediction button,
3. request a prediction,
4. display Bull, Neutral, or Bear,
5. display the model name and version.

---

# 23. Stopping the Deployment

Stop the services with:

```bash
docker compose down
```

This stops and removes the containers while leaving the mounted MLflow database and artifacts on the host.

To rebuild everything after code changes:

```bash
docker compose down
docker compose up --build
```

To run the rebuilt deployment in the background:

```bash
docker compose down
docker compose up --build -d
```

---

# 24. Dashboard Design

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

# 25. Current Deployment Configuration

The current MVP is intentionally fixed to:

| Configuration            | Value                 |
| ------------------------ | --------------------- |
| Asset                    | Bitcoin / BTCUSDT     |
| Candle interval          | 15 minutes            |
| Prediction horizon       | 30 minutes            |
| Prediction classes       | Bull / Neutral / Bear |
| API                      | FastAPI               |
| Dashboard                | Streamlit             |
| Model registry           | MLflow                |
| Container orchestration  | Docker Compose        |
| MLflow host port         | **1060**              |
| MLflow container port    | **1060**              |
| API host port            | **1061**              |
| API container port       | **8000**              |
| Dashboard host port      | **1063**              |
| Dashboard container port | **8501**              |

The API itself has support for multiple horizons, but the current Streamlit dashboard intentionally exposes only the `30m` Bitcoin prediction workflow.

---

# 26. Environment Variables

### API

The Compose environment provides:

```text
MLFLOW_TRACKING_URI=http://mlflow:1060
ROOT_PATH=/api
```

`MLFLOW_TRACKING_URI` tells the API where the deployment MLflow server is located inside the Docker network.

---

### Dashboard

The dashboard receives:

```text
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

# 27. Persistence

The Compose configuration mounts two host directories:

```text
./mlflow:/mlflow
./mlruns:/mlruns
```

This means MLflow's database and artifacts are not stored only inside the temporary container filesystem.

The deployment structure is:

```text
Deployment_test/
│
├── mlflow/
│   └── mlflow.db
│
└── mlruns/
    └── ...
```

Therefore, recreating the MLflow container does not automatically remove these host-side files.

---

# 28. Deployment Artifact

The deployment environment currently contains model artifacts under:

```text
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

The important serialized model artifact is:

```text
model.skops
```

The accompanying environment files describe the dependencies and model configuration associated with that artifact.

The API ultimately loads the model through MLflow rather than manually opening `model.skops`.

---

# 29. Integration Boundaries

Each component has a clearly defined responsibility:

```text
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

# 30. Relationship to the Other Project Components

The broader project is divided into several stages.

```text
Pipeline
    │
    │ Train + evaluate
    ▼
Training MLflow
    │
    │ Model deployment/migration
    ▼
Deployment_test
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
| `Pipeline`                 | Train and evaluate models                 |
| `Online`            | Implement the FastAPI serving layer       |
| `Deployment_test`          | Run the complete containerized deployment |
| `8_CI_CD`                    | Automate build/test/deployment workflows  |
| `9_Monitoring_Observability` | Monitor the deployed system               |

---

# 31. Troubleshooting

## API cannot load a model

Check:

```bash
docker compose logs api
```

Look for errors related to:

* MLflow connectivity,
* model name,
* model version,
* Production stage,
* missing artifacts.

Then check MLflow:

```text
http://35.202.67.240:1060
```

or locally through the SSH tunnel:

```text
http://localhost:1060
```

---

## Dashboard cannot reach API

Check:

```bash
docker compose logs dashboard
```

Inside Docker, the dashboard should use:

```text
http://api:8000
```

not:

```text
http://localhost:1061
```

---

## API is running but prediction returns 503

Check:

```text
http://35.202.67.240:1061/health
```

If the `30m` horizon is not present in `loaded_horizons`, the API does not currently have a Production model available for that horizon.

---

## Containers are running but the dashboard shows an error

Check the dependency chain:

```text
MLflow
  ↓
API
  ↓
Dashboard
```

Then inspect each service:

```bash
docker compose logs mlflow
docker compose logs api
docker compose logs dashboard
```

---

## Docker build fails with "no space left on device"

On the shared VPS, first inspect disk usage:

```bash
df -h
```

Then inspect Docker usage:

```bash
docker system df
```

Do **not** immediately run:

```bash
docker system prune -a --volumes
```

on a shared VPS.

That command can remove unused Docker resources, including volumes, that may belong to other projects or contain important data.

Only clean resources after confirming that they belong to this project and are safe to remove.

---

# 32. Current MVP Scope

The current deployment demonstrates a complete end-to-end ML inference workflow:

```text
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

```text
BTCUSDT
   +
30-minute horizon
   +
Bull / Neutral / Bear
```

This provides a small, testable deployment before expanding the application.

---

# 33. Future Improvements

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

# 34. Important Note

The dashboard presents the output of a machine-learning classification model.

A result such as:

```text
Bull
```

is a model prediction based on the latest available market features. It does not guarantee a future price movement or trading profit.

The application should therefore be understood as an **ML-based decision-support prototype** rather than a guaranteed trading system.

---

# 35. Quick Reference

## Connect to VPS

```bash
ssh eithar@35.202.67.240
```

## Start SSH tunnel — local laptop

Run this in a separate local terminal:

```bash
ssh -N -L 1060:localhost:1060 eithar@35.202.67.240
```

## Verify local MLflow tunnel

```bash
curl http://localhost:1060
```

## Configure local MLflow

```bash
cd ~/Documents/Coding/binance_dashboard
source .venv/bin/activate
export MLFLOW_TRACKING_URI=http://localhost:1060
echo $MLFLOW_TRACKING_URI
```

## Run the training pipeline locally

```bash
python Pipeline/flow.py
```

## Go to deployment directory on VPS

```bash
cd ~/binance_dashboard/Deployment_test
```

## Build and start

```bash
docker compose up --build -d
```

## Check services

```bash
docker compose ps
```

## View logs

```bash
docker compose logs -f
```

## Open VPS dashboard

```text
http://35.202.67.240:1063
```

## Open VPS API

```text
http://35.202.67.240:1061
```

## Open Swagger

```text
http://35.202.67.240:1061/docs
```

## Open VPS MLflow

```text
http://35.202.67.240:1060
```

## Open MLflow locally through SSH tunnel

```text
http://localhost:1060
```

## Stop deployment

```bash
docker compose down
```

---

# Summary

`Deployment_test` is the integration environment that connects the project's trained ML model to a complete online application:

```text
                ┌──────────────┐
                │    MLflow    │
                │    Registry  │
                │    :1060     │
                └──────┬───────┘
                       │
                       ▼
                ┌──────────────┐
                │   FastAPI    │
                │  /predict    │
                │    :8000     │
                └──────┬───────┘
                       │
                       ▼
                ┌──────────────┐
                │  Streamlit   │
                │  Dashboard   │
                │    :8501     │
                └──────┬───────┘
                       │
                       ▼
                     User
```

Docker Compose packages these components into a reproducible environment where model serving, API inference, and the user interface can be tested together.

The current VPS host-port configuration is:

```text
MLflow    → 1060 → container 1060
FastAPI   → 1061 → container 8000
Streamlit → 1063 → container 8501
```

The local development workflow can connect to VPS MLflow through an SSH tunnel:

```text
Local Pipeline
      │
      │ localhost:1060
      ▼
 SSH Tunnel
      │
      ▼
 VPS MLflow:1060
```

The next deployment stage can place the API and Streamlit dashboard behind **Nginx/reverse proxy and HTTPS**, while MLflow can remain privately accessible through the SSH tunnel.
