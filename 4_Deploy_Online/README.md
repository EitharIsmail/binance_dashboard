# Binance Direction Classifier — Online Deployment

This folder contains the **online inference and API serving layer** of the Binance Direction Classifier.

It exposes the trained ML models through a **FastAPI REST API**, retrieves the currently deployed Production models from **MLflow Model Registry**, fetches recent Binance market data, applies the same feature engineering used during training, and returns a directional prediction:

* **Bull** — predicted upward movement
* **Neutral** — predicted movement within the defined threshold
* **Bear** — predicted downward movement

The deployment layer is designed so that the API does **not** contain a separate copy of the trained model. Instead, it loads the version currently promoted to the `Production` stage in MLflow.

---

## 1. Deployment Architecture

The online deployment follows this flow:

```text
                         MLflow Model Registry
                                  │
                                  │
                         Production Model
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────┐
│                     FastAPI Application                   │
│                                                          │
│  Startup                                                │
│    │                                                     │
│    └── Load Production models for supported horizons     │
│                                                          │
│  /predict                                                │
│    │                                                     │
│    ├── Validate requested horizon                         │
│    ├── Fetch recent Binance candles                       │
│    ├── Apply FeatureEngineer                              │
│    ├── Select latest valid feature row                    │
│    ├── Run bundled preprocessing + classifier             │
│    └── Convert prediction to Bull/Neutral/Bear            │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
                     API Response
```

The important separation is:

```text
Training Pipeline                  Online Deployment
─────────────────                  ─────────────────
Historical data                    Recent market data
       ↓                                  ↓
Preprocessing                       Feature engineering
       ↓                                  ↓
Feature engineering                Loaded MLflow model
       ↓                                  ↓
Model training                     Prediction
       ↓                                  ↓
MLflow Registry ────────────────→ Production Model
```

The training pipeline creates and registers the model. The online deployment consumes the model that has been promoted to `Production`.

---

## 2. Responsibilities of This Folder

The deployment layer has five main responsibilities:

1. **Expose an HTTP API**
2. **Load Production models from MLflow**
3. **Retrieve recent Binance market data**
4. **Reproduce the training-time feature engineering**
5. **Return a standardized prediction response**

It does **not** retrain models.

Model training, evaluation, registration, and deployment promotion are handled by [`3_Pipeline`](../3_Pipeline/).

---

## 3. Request-to-Prediction Flow

When a client sends:

```json
{
  "horizon": "30m"
}
```

the API performs the following steps.

### Step 1 — Validate the request

The request is validated using Pydantic.

Supported horizons are:

```text
15m
30m
1h
4h
1d
```

The default horizon is:

```text
30m
```

Currently, the service is configured for:

```text
Symbol: BTCUSDT
Interval: 15m
```

A horizon can only be served if a corresponding model has been trained, registered, and promoted to `Production`.

---

### Step 2 — Select the deployed model

The API maintains an in-memory model state:

```text
horizon → ModelState
```

For example:

```text
30m → BTCUSDT_30m_classifier, Production, v1
```

The model name follows the same convention used by the training pipeline:

```text
{SYMBOL}_{HORIZON}_classifier
```

Examples:

```text
BTCUSDT_15m_classifier
BTCUSDT_30m_classifier
BTCUSDT_1h_classifier
BTCUSDT_4h_classifier
BTCUSDT_1d_classifier
```

This naming convention is important because it provides a direct connection between training and serving.

---

## 4. MLflow Model Loading

Models are loaded from the MLflow Model Registry rather than from a local `.pkl` or `.joblib` file.

The API constructs a model URI equivalent to:

```text
models:/BTCUSDT_30m_classifier/Production
```

The model loader then uses:

```python
mlflow.sklearn.load_model(model_uri)
```

### Why use the registry?

The registry provides a controlled source for the model that should be served.

Instead of:

```text
API → local model file
```

the architecture is:

```text
API → MLflow Registry → Production model
```

This allows the training/deployment workflow to determine which model version is currently Production.

---

## 5. Models Are Loaded at Startup

The API uses FastAPI's lifespan mechanism.

When the application starts:

```text
FastAPI starts
     ↓
load_all_model_states()
     ↓
Check each supported horizon
     ↓
Load Production model if available
     ↓
Store ModelState in memory
```

The API therefore does **not** query MLflow and reload the model for every prediction request.

Conceptually:

```text
Application startup
        │
        ├── Load 15m Production model
        ├── Load 30m Production model
        ├── Load 1h Production model
        ├── Load 4h Production model
        └── Load 1d Production model
                 │
                 ▼
            In-memory state
```

If a horizon has no Production model, that horizon is skipped.

This allows the API to start even when only some horizons have been deployed.

---

## 6. `ModelState`

Each loaded model is represented by a `ModelState` object containing:

| Field              | Purpose                                      |
| ------------------ | -------------------------------------------- |
| `symbol`           | Asset served by the model                    |
| `horizon`          | Prediction horizon                           |
| `model_name`       | MLflow registered model name                 |
| `stage`            | Current registry stage, e.g. `Production`    |
| `pipeline`         | Fitted preprocessing + classifier pipeline   |
| `feature_engineer` | Feature engineering transformer              |
| `version`          | MLflow model version                         |
| `run_id`           | Source MLflow run                            |
| `val_f1`           | Validation Macro F1 recorded during training |
| `test_f1`          | Test Macro F1 recorded during training       |

This means the API knows not only **what model to use**, but also **which registered version it loaded**.

---

# 7. Why Feature Engineering Exists in the API

The model was trained using engineered features rather than raw OHLCV values alone.

During training:

```text
Raw candles
    ↓
FeatureEngineer
    ↓
Engineered features
    ↓
Preprocessing
    ↓
Classifier
```

During online inference, the same logical process must happen:

```text
Recent candles
    ↓
FeatureEngineer
    ↓
Engineered features
    ↓
Loaded preprocessing + classifier
    ↓
Prediction
```

This is essential because the model expects the same feature representation that it saw during training.

The shared implementation is located at:

```text
4_Deploy_Online/shared/feature_engineering.py
```

The deployment Docker image copies this module into the container.

---

## 8. Training/Serving Consistency

One of the important design decisions in this project is to keep feature engineering consistent between training and serving.

The deployment container includes:

```text
shared/
└── feature_engineering.py
```

The same feature engineering logic is used to transform the recent Binance data before prediction.

The trained MLflow artifact also contains the fitted scikit-learn preprocessing and classifier pipeline.

Therefore:

```text
                  Training                         Serving
                  ────────                         ───────
Historical data                              Recent Binance data
      ↓                                             ↓
FeatureEngineer                              FeatureEngineer
      ↓                                             ↓
X features                                    Latest features
      ↓                                             ↓
Preprocessor                                  Loaded preprocessor
      ↓                                             ↓
Classifier                                    Loaded classifier
```

This reduces the risk of **training-serving skew**, where the model receives differently processed data in production than it received during training.

---

# 9. Recent Market Data

The API does not use the historical training dataset for predictions.

Instead, `/predict` obtains recent candles from Binance.

The implementation combines two sources:

### Binance Vision

The API attempts to retrieve the most recent available monthly Binance Vision archive.

Purpose:

* provide a reliable historical base
* maintain consistency with the source used during training
* provide recent finalized candles

### Binance Live REST API

The API also requests the latest candles from:

```text
/api/v3/klines
```

Purpose:

* obtain the newest available market data
* include the currently forming candle
* provide data newer than the latest finalized monthly archive

The two sources are merged:

```text
Binance Vision archive
        +
Binance live REST API
        ↓
Deduplicate by open_time
        ↓
Live data takes precedence
        ↓
Sort chronologically
        ↓
Keep latest 100 candles
```

---

# 10. Why 100 Candles Are Retrieved

The longest feature-engineering window currently used is the 50-period EMA.

The deployment therefore defines:

```python
MIN_CANDLES_FOR_FEATURES = 100
```

This provides a buffer beyond the minimum required history.

The API then:

1. retrieves recent candles,
2. computes all features,
3. removes rows containing feature-generation `NaN` values,
4. selects the most recent valid row.

Conceptually:

```text
100 recent candles
       ↓
Calculate indicators
       ↓
Some early rows may contain NaN
       ↓
Drop invalid rows
       ↓
Take latest valid row
       ↓
Prediction
```

---

# 11. Data Schema Consistency

The API explicitly standardizes incoming Binance data.

The expected schema is:

```text
open_time
open
high
low
close
volume
close_time
quote_asset_volume
number_of_trades
taker_buy_base_asset_volume
taker_buy_quote_asset_volume
ignore
```

The timestamp handling is also standardized:

```text
open_time → datetime
```

while the other fields retain the expected numeric representation.

This is important because the trained preprocessing pipeline expects a compatible input schema and data types.

A mismatch between training and serving schemas can cause prediction-time failures even when the model itself is correct.

---

# 12. Prediction Pipeline

The `/predict` endpoint follows this sequence:

```text
POST /predict
      │
      ▼
Validate horizon
      │
      ▼
Find loaded ModelState
      │
      ▼
Fetch recent BTCUSDT 15m candles
      │
      ▼
FeatureEngineer.transform()
      │
      ▼
Select latest valid row
      │
      ▼
Loaded sklearn Pipeline
      │
      ├── Preprocessor
      │
      └── Classifier
      │
      ▼
Mapped class prediction
      │
      ▼
Original label
      │
      ▼
Bull / Neutral / Bear
```

---

# 13. Label Mapping

The training pipeline uses:

```text
Original labels:

-1 → Bear
 0 → Neutral
 1 → Bull
```

Machine-learning libraries such as XGBoost, LightGBM, and CatBoost require contiguous non-negative class labels, so the training pipeline maps them to:

```text
-1 → 0
 0 → 1
 1 → 2
```

The deployment layer reverses this mapping:

```text
0 → -1 → Bear
1 →  0 → Neutral
2 →  1 → Bull
```

This ensures that the class returned by the API corresponds to the original target definition used during training.

---

# 14. API Endpoints

## `GET /`

Returns basic service information.

Example:

```json
{
  "message": "Binance Direction Classifier API is running",
  "docs": "/docs",
  "health": "/health",
  "predict": "/predict"
}
```

---

## `GET /health`

Reports whether models have been successfully loaded.

Example:

```json
{
  "status": "ok",
  "symbol": "BTCUSDT",
  "loaded_horizons": {
    "30m": {
      "model_name": "BTCUSDT_30m_classifier",
      "version": "1",
      "model_alias": "Production"
    }
  }
}
```

This endpoint is useful for checking:

* whether the API is running,
* which symbol is configured,
* which horizons currently have Production models,
* which model version is being served.

---

## `POST /predict`

Generates a directional prediction.

Request:

```json
{
  "horizon": "30m"
}
```

Example response:

```json
{
  "prediction": "Bull",
  "model_name": "BTCUSDT_30m_classifier",
  "model_version": "1"
}
```

The response identifies both the prediction and the exact registered model version used to generate it.

---

# 15. Swagger / OpenAPI Documentation

FastAPI automatically provides interactive API documentation.

After starting the application, open:

```text
http://127.0.0.1:8000/docs
```

The Swagger UI allows you to:

1. inspect available endpoints,
2. inspect request/response schemas,
3. select `POST /predict`,
4. enter a horizon,
5. execute the request,
6. inspect the API response.

For example:

```json
{
  "horizon": "30m"
}
```

---

# 16. Error Handling

The API distinguishes several deployment situations.

### No model deployed for a requested horizon

Returns HTTP `503`.

Example:

```text
No model currently deployed for horizon='1h'
```

This means the API itself may be running, but no Production model exists for that horizon.

### Model unavailable

If a loaded model pipeline is missing, the API returns HTTP `503`.

### Insufficient feature history

If enough valid candles are not available to compute the required features, the API returns HTTP `503`.

### Unexpected prediction failure

Unexpected exceptions are logged and returned as HTTP `500`.

The application also logs exceptions so that operational failures can be investigated.

---

# 17. Docker Deployment

The API is containerized using:

```text
api/Dockerfile
```

The container is based on:

```text
python:3.13-slim
```

The Docker image copies:

```text
shared/
api/
```

and installs the API dependencies from:

```text
api/requirements.txt
```

The container exposes:

```text
8000
```

and starts:

```text
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

The resulting architecture is:

```text
Host
  │
  ▼
Docker container
  │
  └── FastAPI
        │
        ├── shared FeatureEngineer
        ├── MLflow client
        └── Binance API/Vision
```

---

# 18. Running the API Locally

From this directory:

```bash
cd 4_Deploy_Online
```

start FastAPI with:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/health
```

---

# 19. Testing the API

Using `curl`:

```bash
curl -X POST \
  http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"horizon":"30m"}'
```

Example:

```json
{
  "prediction": "Bull",
  "model_name": "BTCUSDT_30m_classifier",
  "model_version": "1"
}
```

The prediction itself can change as new market data arrives. The model name and version identify which registered model generated the response.

---

# 20. Project Structure

The source structure is:

```text
4_Deploy_Online/
├── api/
│   ├── config.py
│   ├── Dockerfile
│   ├── main.py
│   ├── model_loader.py
│   ├── requirements.txt
│   └── schema.py
│
├── models/
│
├── logs/
│
├── shared/
│   └── feature_engineering.py
│
└── README.md
```

### `api/main.py`

The main FastAPI application.

Responsible for:

* application startup/shutdown,
* model loading,
* health checks,
* prediction requests,
* recent market-data retrieval,
* feature transformation,
* prediction response formatting,
* error handling.

### `api/model_loader.py`

Responsible for:

* connecting to MLflow,
* identifying registered model names,
* loading Production models,
* retrieving model metadata,
* constructing `ModelState` objects.

### `api/schema.py`

Defines the API contract using Pydantic and Python enums.

Contains:

```text
Horizon
CoinRequest
PredictionResponse
```

### `api/config.py`

Contains configuration shared by the API, including:

* data configuration,
* target configuration,
* MLflow configuration.

### `shared/feature_engineering.py`

Contains the feature engineering logic required during inference.

Keeping this logic available to the serving layer helps maintain consistency with the training pipeline.

### `api/Dockerfile`

Defines the container image and startup command for the FastAPI service.

### `logs/`

Runtime logs generated by the application.

### `models/`

Reserved for model-related deployment assets. The current serving design loads the actual Production model from MLflow rather than depending on a local model file.

---

# 21. Configuration Relationship

The deployment layer depends on three important pieces of configuration:

```text
DataConfig
     │
     ├── Symbol
     ├── Interval
     └── Binance Vision configuration
     
TargetConfig
     │
     └── Supported horizons

MLflowConfig
     │
     └── MLflow tracking URI
```

The API uses the configured symbol and supported horizons when loading models.

The current deployment configuration is centered on:

```text
Symbol: BTCUSDT
Candle interval: 15m
Supported horizons: 15m, 30m, 1h, 4h, 1d
```

Supported does not necessarily mean deployed.

For example:

```text
Supported horizon: 1h
        ≠
Production model available: 1h
```

A horizon becomes actually servable only after its corresponding model has been promoted to `Production`.

---

# 22. Deployment Lifecycle

The broader project separates **model creation** from **model serving**.

```text
3_Pipeline
     │
     │ Train candidate models
     ▼
MLflow Training Registry
     │
     │ Deployment gate
     ▼
Selected Production candidate
     │
     │ Migration / deployment process
     ▼
Deployment MLflow Registry
     │
     ▼
Production
     │
     ▼
4_Deploy_Online
     │
     ▼
FastAPI
     │
     ▼
Prediction
```

This separation means that the API does not decide which newly trained model should become Production.

The model lifecycle is handled before the serving layer loads the model.

---

# 23. Why the API Loads Models at Startup

Loading models once at startup provides a simple serving pattern:

```text
Startup:
    MLflow → load model → memory

Request:
    request → in-memory model → prediction
```

rather than:

```text
Every request:
    request
       ↓
    MLflow
       ↓
    download model
       ↓
    load model
       ↓
    prediction
```

The first approach avoids repeatedly loading the model for every request and keeps the prediction path focused on obtaining current data, computing features, and performing inference.

---

# 24. Current MVP Deployment

The current online MVP has:

| Component            | Current configuration |
| -------------------- | --------------------- |
| Asset                | BTCUSDT               |
| Candle interval      | 15 minutes            |
| Default horizon      | 30 minutes            |
| Prediction classes   | Bull / Neutral / Bear |
| API framework        | FastAPI               |
| Model source         | MLflow Model Registry |
| Served model stage   | Production            |
| Containerization     | Docker                |
| API port             | 8000                  |
| Interactive API docs | Swagger / OpenAPI     |

At the current deployment state, a Production model is available for the `30m` horizon. Other horizons can be supported by the API once corresponding models are trained and promoted to Production.

---

# 25. Relationship with the Training Pipeline

The two folders have different responsibilities.

| `3_Pipeline`                | `4_Deploy_Online`           |
| --------------------------- | --------------------------- |
| Acquire historical data     | Fetch recent data           |
| Clean data                  | Standardize live data       |
| Create target               | No target creation          |
| Engineer training features  | Engineer inference features |
| Split train/validation/test | No model training           |
| Train candidate models      | Load trained model          |
| Evaluate models             | Generate predictions        |
| Track experiments           | Serve predictions           |
| Register models             | Consume Production model    |
| Apply deployment gate       | Expose API                  |

The overall workflow is therefore:

```text
Historical Binance Data
        ↓
3_Pipeline
        ↓
Train + Evaluate
        ↓
MLflow Registry
        ↓
Deployment Gate
        ↓
Production Model
        ↓
4_Deploy_Online
        ↓
FastAPI
        ↓
Recent Binance Data
        ↓
Features
        ↓
Prediction
```

---

# 26. Important Design Considerations

### Training and serving must use compatible data schemas

The API deliberately standardizes Binance data before feature engineering.

This prevents issues caused by differences such as:

```text
Training:
open_time = datetime64

Serving:
open_time = integer
```

or unexpected differences in numeric columns.

---

### Feature engineering must remain consistent

Changing the feature definitions in the training pipeline without updating the serving implementation can cause the model to receive an incompatible feature set.

The shared feature engineering module helps reduce this risk.

---

### The model artifact contains preprocessing

The trained MLflow artifact is a fitted scikit-learn pipeline containing:

```text
Preprocessor
     +
Classifier
```

Therefore the API does not need to manually recreate the training-time scaler and imputer.

---

### Production is versioned

The API response exposes:

```json
{
  "model_name": "...",
  "model_version": "..."
}
```

This provides basic prediction traceability: a prediction can be associated with the exact registered model version that generated it.

---

# 27. Limitations and Future Improvements

The current deployment is an MVP. Potential improvements include:

* support for multiple cryptocurrency symbols,
* deployment of additional horizons,
* automated model promotion/migration,
* more robust model health checks,
* request and response monitoring,
* prediction logging,
* data-quality monitoring,
* feature drift detection,
* model-performance monitoring,
* scheduled model refresh/retraining,
* authentication and authorization,
* rate limiting,
* API metrics and observability,
* production-grade secrets/configuration management,
* container orchestration,
* automated CI/CD deployment.

For a trading-oriented evaluation, additional backtesting would also be required to assess factors such as transaction costs, slippage, and strategy-level performance. Classification accuracy alone does not establish profitability.

---

# 28. Important Note

This service provides **machine-learning predictions of directional market movement** based on recent Bitcoin market data.

A prediction such as:

```text
Bull
```

means that the deployed classifier assigned the latest feature vector to the Bull class according to the target definition used during training. It does not guarantee that the market will move upward or that a trading decision will be profitable.

The system should therefore be understood as an **ML decision-support prototype**, not as a guarantee of future market performance.

---

## Related Components

* [`3_Pipeline`](../3_Pipeline/) — historical data processing, feature engineering, model training, evaluation, MLflow tracking, and model registration.
* [`7_Deployment_Test`](../7_Deployment_Test/) — integration testing of the deployed API and surrounding services.
* [`8_CI_CD`](../8_CI_CD/) — continuous integration/deployment workflows.
* [`9_Monitoring_Observability`](../9_Monitoring_Observability/) — monitoring and observability components.

---

## Summary

The online deployment layer turns the trained Binance direction classifier into a reusable prediction service:

```text
MLflow Production Model
          +
Recent Binance Market Data
          ↓
Shared Feature Engineering
          ↓
Fitted Preprocessing
          ↓
Classifier
          ↓
Bull / Neutral / Bear
          ↓
FastAPI JSON Response
```

Its main design goal is to keep **model lifecycle management, data processing, and online inference clearly separated** while maintaining consistency between the training and serving environments.
