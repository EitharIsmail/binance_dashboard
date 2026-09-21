# Binance Direction Classifier

> **An end-to-end MLOps system for short-term Bitcoin market direction prediction.**

## Overview

Trading on cryptocurrency exchanges such as Binance involves processing a large amount of constantly changing market information. Price charts, candlestick patterns, trading volume, volatility, momentum indicators, and historical price movements all provide signals that may help understand the current market.

The challenge is not simply accessing this information — it is **turning a large amount of market data into a timely and understandable signal**.

When deciding whether to buy, sell, or wait, a trader may need to inspect multiple indicators and timeframes and interpret how they interact. Doing this manually can be time-consuming, especially when the market is changing continuously.

This project explores how machine learning can help reduce that information overload by providing a simple, automated prediction of the **expected short-term price direction**.

---

# The Problem

When trading Bitcoin on Binance, users are exposed to a large amount of information:

* Price movements
* Candlestick data
* Trading volume
* Short- and long-term trends
* Volatility
* Momentum indicators
* Historical market behavior
* Different timeframes

To make a trading decision, a user may need to combine many of these signals and repeatedly monitor the market.

The core problem addressed by this project is:

> **Can machine learning transform recent Binance market information into a simple short-term directional signal that is easier to interpret than manually analyzing many indicators?**

Instead of requiring the user to interpret every indicator individually, the system produces one of three directional classes:

| Prediction    | Meaning                                                |
| ------------- | ------------------------------------------------------ |
| 🟢 **Bull**   | The model predicts upward price movement               |
| ⚪ **Neutral** | The model predicts no significant directional movement |
| 🔴 **Bear**   | The model predicts downward price movement             |

**Important:** This project is a machine-learning prediction system, not financial advice and not an automated trading system. A prediction represents what the trained model estimates from historical market patterns; it does not guarantee future price movement.

---

# The Solution

The solution is an **end-to-end MLOps pipeline** that automatically transforms Binance market data into a deployable machine-learning prediction service.

The current version is intentionally an **MVP (Minimum Viable Product)**.

### Current MVP scope

* **Asset:** Bitcoin (`BTCUSDT`)
* **Prediction horizon:** 30 minutes
* **Prediction:** Bull / Neutral / Bear
* **Data source:** Binance market data
* **Machine learning:** Multiple classification algorithms are evaluated
* **Experiment tracking:** MLflow
* **Model registry:** MLflow Model Registry
* **Training orchestration:** Prefect
* **Online serving:** FastAPI
* **Containerization:** Docker
* **User interface:** Streamlit
* **Deployment:** Docker Compose

The system is designed so that this initial MVP can be expanded without redesigning the entire architecture.

Future versions can introduce additional cryptocurrencies, prediction horizons, features, models, monitoring, and eventually more sophisticated decision-support capabilities.

---

# How It Works

At a high level, the system follows this process:

```text
                    Binance Market Data
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Data Acquisition  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Data Preprocessing  │
                 │ Cleaning & Validation│
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Feature Engineering │
                 │ Price / Volume /     │
                 │ Momentum / Volatility│
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Model Training      │
                 │ Multiple Classifiers│
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Model Evaluation    │
                 │ Accuracy / F1       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ MLflow Tracking &   │
                 │ Model Registry      │
                 └──────────┬──────────┘
                            │
                       Production
                            │
                            ▼
                 ┌─────────────────────┐
                 │ FastAPI Prediction  │
                 │ Service             │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Streamlit Dashboard │
                 └─────────────────────┘
```

The important separation is between **offline training** and **online serving**.

Training can be computationally expensive and does not need to happen every time a user requests a prediction. Serving, on the other hand, needs to be lightweight and continuously available.

Therefore:

```text
Training
────────
Acquire → Clean → Features → Train → Evaluate → Register
                                      │
                                      ▼
                               MLflow Registry
                                      │
                                      ▼
                                  Production


Serving
───────
User → Dashboard → FastAPI → Production Model → Prediction
```

---

# MLOps Architecture

The project is divided into independent stages.

| Component                                                      | Purpose                                                                                                          |
| -------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| [`3_Pipeline/`](./3_Pipeline/README.md)                        | Offline data processing, feature engineering, model training, evaluation, MLflow tracking and model registration |
| [`4_Deploy_Online/`](./4_Deploy_Online/README.md)              | Online FastAPI inference service that loads the Production model and serves predictions                          |
| [`7_Deployment_Test/`](./7_Deployment_Test/)                   | Docker Compose environment used to test the complete deployment architecture                                     |
| [`8_CI_CD/`](./8_CI_CD/)                                       | Continuous integration and deployment components                                                                 |
| [`9_Monitoring_Observability/`](./9_Monitoring_Observability/) | Monitoring and observability components                                                                          |

---

# Training vs. Serving

A central design principle of this project is:

> **Training is expensive and infrequent; inference is lightweight and continuous.**

The training pipeline does not need to run every time a prediction is requested.

Instead, the training system produces a versioned model and stores it in MLflow.

The online API then retrieves the model currently marked as **Production**.

```text
┌──────────────────────┐
│   Training Pipeline  │
│                      │
│ Data → Features →    │
│ Models → Evaluation  │
└──────────┬───────────┘
           │
           │ Register
           ▼
┌──────────────────────┐
│   MLflow Registry    │
│                      │
│ Model versions       │
│ Metrics              │
│ Production status    │
└──────────┬───────────┘
           │
           │ Load Production
           ▼
┌──────────────────────┐
│     FastAPI          │
│                      │
│ Live inference       │
└──────────────────────┘
```

This separation means that deploying a new model does not require rewriting the API. The API simply loads the model selected by the model registry.

---

# Model Per Prediction Horizon

The project treats each prediction horizon as a separate machine-learning problem.

For example:

```text
BTCUSDT + 15m → one model
BTCUSDT + 30m → another model
BTCUSDT + 1h   → another model
BTCUSDT + 4h   → another model
BTCUSDT + 1d   → another model
```

The reason is that predicting whether Bitcoin will move during the next 30 minutes is fundamentally different from predicting its movement over the next 4 hours.

Different horizons can have:

* Different target distributions
* Different levels of market noise
* Different relevant features
* Different optimal model parameters
* Different model performance

Models therefore follow the naming convention:

```text
{symbol}_{horizon}_classifier
```

For example:

```text
BTCUSDT_30m_classifier
```

The current MVP uses the **30-minute model**.

---

# Current MVP

The current production workflow focuses on:

```text
Asset       → BTCUSDT
Horizon     → 30 minutes
Classes     → Bull / Neutral / Bear
Model       → AdaBoost classifier
Serving     → FastAPI
UI          → Streamlit
Registry    → MLflow
Containers  → Docker Compose
```

The MVP is intentionally limited in scope so that the complete MLOps lifecycle can be developed and validated before expanding the system.

---

# Features

The model uses market-derived features based on recent price and volume behavior.

| Feature            | What it measures                        | Simple meaning                                            |
| ------------------ | --------------------------------------- | --------------------------------------------------------- |
| `return_1`         | Price change over 1 candle              | How much price changed recently                           |
| `return_3`         | Price change over 3 candles             | Short-term accumulated price movement                     |
| `volatility_10`    | Standard deviation of recent returns    | How much price has been fluctuating                       |
| `ema_20`           | 20-period exponential moving average    | Shorter-term trend                                        |
| `ema_50`           | 50-period exponential moving average    | Longer-term trend                                         |
| `dist_from_ema_20` | Current price relative to EMA-20        | Whether price is above or below its short-term trend      |
| `rsi_14`           | Relative Strength Index over 14 periods | Recent momentum between buying and selling pressure       |
| `volume_sma_20`    | Average volume over 20 periods          | What normal recent trading activity looks like            |
| `volume_ratio`     | Current volume ÷ average volume         | Whether current trading activity is unusually high or low |

These features are not intended to represent every possible market signal. They provide a starting feature set for the MVP and can be extended as the project evolves.

---

# Machine Learning Models

The training pipeline evaluates multiple classification algorithms rather than assuming that one model will always perform best.

Current model candidates include:

* Logistic Regression
* Random Forest
* Extra Trees
* Gradient Boosting
* AdaBoost
* Support Vector Classifier
* K-Nearest Neighbors
* XGBoost
* LightGBM
* CatBoost

Models are evaluated using metrics such as:

* Accuracy
* Macro F1-score

The selected model is then registered in MLflow and can be promoted to the Production stage through the deployment process.

---

# Target Definition

The model predicts the **direction of future price movement**, rather than the exact future Bitcoin price.

The target is based on the future price movement relative to a configurable threshold.

The three classes are:

```text
-1 → Bear
 0 → Neutral
 1 → Bull
```

This makes the problem a **three-class classification problem**.

The distinction between Bull, Neutral, and Bear is useful because a small price movement may not represent a meaningful directional change.

---

# Technology Stack

| Technology                        | Role                                   |
| --------------------------------- | -------------------------------------- |
| **Python**                        | Core programming language              |
| **Pandas / NumPy**                | Data processing                        |
| **Scikit-learn**                  | Machine learning                       |
| **XGBoost / LightGBM / CatBoost** | Additional ML algorithms               |
| **Prefect**                       | Pipeline orchestration                 |
| **MLflow**                        | Experiment tracking and model registry |
| **FastAPI**                       | Online prediction API                  |
| **Streamlit**                     | Interactive dashboard                  |
| **Docker**                        | Containerization                       |
| **Docker Compose**                | Multi-service deployment               |
| **Git / GitHub**                  | Version control and collaboration      |

---

# Online Deployment Architecture

The current deployment consists of three main services:

```text
                         Browser
                            │
                            ▼
                  ┌──────────────────┐
                  │    Streamlit     │
                  │    Dashboard     │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │     FastAPI      │
                  │   Prediction API │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │      MLflow      │
                  │ Model Registry   │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Production Model │
                  │   BTCUSDT 30m    │
                  └──────────────────┘
```

Docker Compose manages these services and provides the networking between them.

---

# API Example

The online API accepts a prediction horizon:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"horizon": "30m"}'
```

Example response:

```json
{
  "prediction": "Bull",
  "model_name": "BTCUSDT_30m_classifier",
  "model_version": "1"
}
```

The API therefore provides a simple interface between the machine-learning system and any application that wants to consume its predictions.

---

# Model Lifecycle

The project follows a model lifecycle rather than simply saving a `.pkl` file.

```text
Train
  │
  ▼
Evaluate
  │
  ▼
Track experiment in MLflow
  │
  ▼
Register model
  │
  ▼
Compare candidate with deployed model
  │
  ▼
Promote to Production
  │
  ▼
FastAPI loads Production model
  │
  ▼
Serve predictions
```

This makes model versions, metrics, and deployment state traceable.

---

# Project Structure

```text
binance_dashboard/
│
├── 3_Pipeline/
│   ├── config/
│   ├── data/
│   ├── models/
│   ├── src/
│   ├── flow.py
│   ├── main.py
│   └── README.md
│
├── 4_Deploy_Online/
│   ├── api/
│   ├── shared/
│   └── README.md
│
├── 7_Deployment_Test/
│   ├── dashboard/
│   ├── mlflow/
│   ├── mlruns/
│   └── docker-compose.yml
│
├── 8_CI_CD/
│
├── 9_Monitoring_Observability/
│
├── dashboard.py
├── Dockerfile
├── requirements.txt
├── pyproject.toml
├── migrate_model.py
└── README.md
```

---

# Quick Start

## 1. Train the models

```bash
cd 3_Pipeline
python main.py --all-horizons
```

This runs the offline training workflow and produces tracked and registered models.

## 2. Start the online deployment

From the deployment test environment:

```bash
cd ../7_Deployment_Test
sudo docker compose up --build
```

Or run it in the background:

```bash
sudo docker compose up --build -d
```

## 3. Check the services

```bash
sudo docker compose ps
```

The main services are:

```text
MLflow      → localhost:5000
FastAPI     → localhost:8000
Dashboard   → localhost:8501
```

## 4. Test the API

Health check:

```bash
curl http://localhost:8000/health
```

Prediction:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"horizon":"30m"}'
```

## 5. Open the dashboard

Open:

```text
http://localhost:8501
```

---

# MVP → Future Development

The current implementation is deliberately a starting point rather than the final system.

Planned extensions include:

### Market coverage

* Multiple cryptocurrencies
* Additional trading pairs
* More market data sources

### Prediction horizons

* 15 minutes
* 30 minutes
* 1 hour
* 4 hours
* 1 day
* Additional horizons where appropriate

### Feature engineering

Potential additions include:

* RSI 7 / 14 / 50
* EMA 20 / 50 / 200 relationships
* MACD
* ATR / NATR
* Bollinger Bands
* Additional volume indicators
* More sophisticated price and volatility features

### Machine learning

Future development may explore:

* Hyperparameter optimization
* Feature selection
* Ensemble methods
* Time-series-specific approaches
* Deep learning models
* Model calibration
* Probability-based predictions

### MLOps

The project can be extended with:

* Automated CI/CD
* Scheduled retraining
* Data drift detection
* Model performance monitoring
* Prediction monitoring
* Logging and observability
* Automated model promotion workflows
* Model rollback

### Product development

The longer-term goal is to make the prediction system more useful as a **decision-support tool**, while keeping the machine-learning prediction and the user's eventual trading decision clearly separated.

---

# Project Philosophy

This project is not designed around the idea of simply training a model and obtaining a high accuracy score.

The main objective is to demonstrate the **complete machine-learning lifecycle**:

```text
Problem
   ↓
Data
   ↓
Features
   ↓
Experimentation
   ↓
Model Training
   ↓
Evaluation
   ↓
Model Registry
   ↓
Deployment
   ↓
Inference
   ↓
Monitoring
   ↓
Continuous Improvement
```

The project therefore combines **machine learning, software engineering, and MLOps** into one deployable system.

---

# Disclaimer

This project is intended for **educational, research, and software-engineering purposes**.

Cryptocurrency markets are highly volatile and unpredictable. The model's Bull / Neutral / Bear prediction is an estimate generated from historical and recent market data and should not be interpreted as a guarantee of future price movement or as financial advice.

The system currently provides predictions only; it does **not automatically execute trades or place buy/sell orders on Binance**.
