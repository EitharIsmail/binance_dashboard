# Binance ML Pipeline

> **Offline data, feature engineering, model training, evaluation, experiment tracking, and model promotion pipeline for Binance market-direction prediction.**

This directory contains the **offline machine-learning pipeline** for the Binance Direction Classifier project.

Its responsibility is to transform historical Binance market data into a trained, evaluated, versioned, and potentially deployable machine-learning model.

The pipeline is orchestrated with **Prefect** and uses **MLflow** for experiment tracking and model registry operations.

---

# Pipeline Objective

The online application needs a trained model that can answer a question such as:

> **Given recent Bitcoin market behavior, what is the predicted direction of the price over the next 30 minutes?**

The offline pipeline prepares that model.

At a high level:

```text
Binance Historical Data
        │
        ▼
Data Acquisition
        │
        ▼
Data Validation & Cleaning
        │
        ▼
Target Creation
        │
        ▼
Feature Engineering
        │
        ▼
Chronological Train / Validation / Test Split
        │
        ▼
Preprocessing
        │
        ▼
Train Multiple Models
        │
        ▼
Validation Model Selection
        │
        ▼
Test Evaluation
        │
        ▼
MLflow Experiment Tracking
        │
        ▼
MLflow Model Registry
        │
        ▼
Staging / Production Decision
```

The pipeline is intentionally separated from online serving.

**Training happens offline; prediction happens online.**

---

# Why an Offline Pipeline?

Training a machine-learning model is computationally more expensive than making a single prediction.

Instead of training a model every time a user requests a prediction, this project follows a model lifecycle:

```text
Train
  ↓
Evaluate
  ↓
Register
  ↓
Promote
  ↓
Serve
```

The resulting model is stored in MLflow and can later be loaded by the online FastAPI service.

This separation also makes experiments reproducible and allows different versions of a model to be tracked over time.

---

# Current MVP Configuration

The current MVP focuses on Bitcoin and a short-term prediction horizon.

| Configuration              | Current value         |
| -------------------------- | --------------------- |
| Trading pair               | `BTCUSDT`             |
| Candle interval            | `15m`                 |
| Default prediction horizon | `30m`                 |
| Target threshold           | `0.002` / 0.2%        |
| Validation set             | 15%                   |
| Test set                   | 15%                   |
| Training set               | 70%                   |
| Prediction classes         | Bear / Neutral / Bull |
| Experiment tracker         | MLflow                |
| Orchestrator               | Prefect               |

The pipeline is configurable, so other symbols, date ranges, and horizons can be supplied through the command line.

---

# Pipeline Architecture

The Prefect flow consists of six main stages:

```text
1. acquire-data
        ↓
2. preprocess-data
        ↓
3. engineer-features
        ↓
4. align-and-save
        ↓
5. train-and-evaluate
        ↓
6. deploy-model
```

Each stage is implemented as a Prefect task.

The main flow is defined in:

```text
flow.py
```

The command-line entry point is:

```text
main.py
```

---

# 1. Data Acquisition

### File

```text
src/Data/data_acquisition.py
```

### Class

```python
BinanceDataAcquisition
```

The acquisition stage downloads historical market data from **Binance Vision**.

Instead of requesting individual candles one by one, the pipeline constructs monthly Binance Vision archive URLs.

For example:

```text
BTCUSDT + 15m + January 2023
        ↓
BTCUSDT-15m-2023-01.zip
```

The downloaded ZIP contains the historical candlestick CSV data.

### Data flow

```text
Binance Vision
      │
      ▼
Monthly ZIP files
      │
      ▼
CSV
      │
      ▼
Pandas DataFrame
      │
      ▼
Timestamp/type processing
      │
      ▼
Monthly DataFrames
      │
      ▼
Concatenation
      │
      ▼
Sort + deduplicate
      │
      ▼
Parquet
```

The resulting raw dataset is stored under:

```text
data/raw/
```

Example:

```text
data/raw/BTCUSDT_15m_raw.parquet
```

### Data columns

The Binance candlestick data contains:

* `open_time`
* `open`
* `high`
* `low`
* `close`
* `volume`
* `close_time`
* `quote_asset_volume`
* `number_of_trades`
* `taker_buy_base_asset_volume`
* `taker_buy_quote_asset_volume`
* `ignore`

The acquisition stage also handles Binance timestamp formats and converts the timestamps into Pandas datetime values.

---

# 2. Data Preprocessing

### File

```text
src/Data/data_preprocessing.py
```

### Class

```python
DataPreprocessor
```

This stage performs ETL, validation, cleaning, and **target creation**.

The order is deliberate:

```text
Validate
   ↓
Sort
   ↓
Deduplicate
   ↓
Handle missing values
   ↓
Create target
   ↓
Remove rows without future labels
```

---

## Data Validation

The pipeline verifies that:

* `open_time` is a valid datetime
* numerical market columns are numeric
* timestamps are converted correctly

It also contains logic for detecting whether timestamps are represented in milliseconds or microseconds.

---

## Chronological Ordering

Financial time-series data must remain ordered by time.

The pipeline sorts by:

```text
open_time
```

and removes duplicate timestamps.

This is important because later stages use previous and future observations.

---

## Missing Values

The pipeline handles missing values using **forward filling**:

```python
df[self.core_cols] = df[self.core_cols].ffill()
```

Forward filling uses information from earlier observations rather than future observations.

This helps avoid introducing future information into historical rows.

---

# Target Creation

The target is created during preprocessing rather than feature engineering.

This is an important design decision.

The target depends only on the current and future `close` prices:

```text
Current close
      │
      │
      └──────────────► Future close
                           │
                           ▼
                     Future return
                           │
                           ▼
                    Bull / Neutral / Bear
```

The future return is:

```text
future_return =
    (future_close - current_close) / current_close
```

For the default configuration:

```text
interval = 15m
horizon  = 30m
```

Therefore:

```text
30 minutes / 15 minutes = 2 periods
```

The pipeline uses:

```python
future_close = close.shift(-2)
```

---

# Target Classes

The default threshold is:

```text
0.002 = 0.2%
```

The target is defined as:

|          Future return | Class | Meaning |
| ---------------------: | ----: | ------- |
|            `>= +0.002` |   `1` | Bull    |
|            `<= -0.002` |  `-1` | Bear    |
| Between the thresholds |   `0` | Neutral |

Therefore:

```text
-1 → Bear
 0 → Neutral
 1 → Bull
```

Rows at the end of the dataset that do not have enough future data to calculate the target are removed.

### Why?

Suppose the final candle is:

```text
2024-01-01 23:45
```

There is no candle 30 minutes into the future inside the dataset.

Therefore its target cannot be calculated and the row cannot be used for supervised training.

---

# Horizon Validation

The preprocessing class validates the relationship between the candle interval and prediction horizon.

For example:

```text
15m interval + 30m horizon
        ✓ valid

15m interval + 1h horizon
        ✓ valid

15m interval + 45m horizon
        ✓ valid

15m interval + 20m horizon
        ✗ invalid
```

The horizon must be:

1. Positive
2. At least as large as the candle interval
3. An exact multiple of the candle interval

This guarantees that the future target can be represented as a whole number of candles.

---

# 3. Feature Engineering

### File

```text
src/Features/feature_engineering.py
```

### Class

```python
FeatureEngineer
```

Feature engineering transforms raw market information into numerical signals that the machine-learning models can use.

The current MVP uses price, momentum, trend, volatility, and volume-based features.

```text
Raw OHLCV data
      │
      ▼
Feature Engineering
      │
      ├── Returns
      ├── Volatility
      ├── EMA
      ├── RSI
      └── Volume indicators
      │
      ▼
Feature matrix X
```

---

# Current Features

| Feature            | Calculation / Meaning                                    |
| ------------------ | -------------------------------------------------------- |
| `return_1`         | Percentage change over one candle                        |
| `return_3`         | Percentage change over three candles                     |
| `volatility_10`    | Rolling standard deviation of `return_1` over 10 candles |
| `ema_20`           | 20-period exponential moving average                     |
| `ema_50`           | 50-period exponential moving average                     |
| `dist_from_ema_20` | Relative distance between price and EMA-20               |
| `rsi_14`           | 14-period Relative Strength Index                        |
| `volume_sma_20`    | 20-period rolling average volume                         |
| `volume_ratio`     | Current volume relative to average volume                |

---

# Why Feature Engineering Comes Before Model Training

The raw candle data contains information such as:

```text
open
high
low
close
volume
```

A machine-learning model can use these values directly, but derived features can provide additional representations of recent market behavior.

For example:

```text
close
 │
 ├── return
 ├── EMA
 ├── distance from EMA
 ├── RSI
 └── volatility
```

The goal is to provide the models with multiple numerical descriptions of recent price and volume behavior.

---

# Feature Warm-up and Alignment

Some technical indicators require historical observations before they produce valid values.

For example:

```text
EMA-50
```

needs an initial period to become established.

Similarly, rolling indicators such as:

```text
volatility_10
volume_sma_20
```

produce NaN values at the beginning.

Therefore, after feature engineering, the pipeline performs:

```text
Feature Engineering
        ↓
Rows containing NaN
        ↓
Remove invalid warm-up rows
        ↓
Reset X/y indexes
```

This is handled by:

```python
align_features_and_target()
```

The target is aligned with exactly the same rows that remain in the feature matrix.

---

# 4. Train / Validation / Test Split

### File

```text
src/Models/model_training.py
```

The pipeline uses a **chronological split**, not a random split.

Default:

```text
70% → Training
15% → Validation
15% → Test
```

The order is preserved:

```text
Past                                              Future
│                                                   │
├──────────────┬─────────────┬─────────────────────┤
│    Train     │ Validation  │        Test         │
│     70%      │     15%     │         15%         │
└──────────────┴─────────────┴─────────────────────┘
```

### Why no shuffling?

Financial time-series observations are temporally related.

Randomly mixing observations could allow information from later periods to influence earlier training decisions.

A chronological split better represents the real deployment situation:

> Train on the past → validate on a later period → test on an even later unseen period.

---

# 5. Preprocessing Before Modeling

The model-training pipeline automatically selects numeric features.

The preprocessing pipeline is:

```text
Numerical Features
       │
       ▼
SimpleImputer
(median)
       │
       ▼
StandardScaler
       │
       ▼
Machine Learning Model
```

The preprocessor is **fitted only on the training set**.

```text
X_train
   │
   └── fit + transform

X_validation
   │
   └── transform only

X_test
   │
   └── transform only
```

This prevents information from the validation or test sets from influencing the preprocessing parameters.

---

# 6. Model Benchmarking

The pipeline evaluates multiple machine-learning algorithms.

Current candidates:

| Model               | General approach                                      |
| ------------------- | ----------------------------------------------------- |
| Logistic Regression | Linear classification model                           |
| Random Forest       | Ensemble of randomized decision trees                 |
| Extra Trees         | Highly randomized tree ensemble                       |
| XGBoost             | Gradient-boosted decision trees                       |
| LightGBM            | Efficient gradient boosting                           |
| CatBoost            | Gradient boosting designed to handle complex patterns |
| Gradient Boosting   | Sequentially improved decision trees                  |
| AdaBoost            | Sequentially focuses on difficult observations        |
| SVC (RBF)           | Kernel-based classifier                               |
| KNN                 | Classifies using nearby observations                  |

The objective is not to assume that one algorithm is universally suitable.

Instead:

```text
Same training data
       │
       ├── Logistic Regression
       ├── Random Forest
       ├── Extra Trees
       ├── XGBoost
       ├── LightGBM
       ├── CatBoost
       ├── Gradient Boosting
       ├── AdaBoost
       ├── SVC
       └── KNN
              │
              ▼
        Compare validation
           performance
```

---

# Class Balancing

The training data uses balanced sample weights:

```python
compute_sample_weight("balanced", y_train)
```

This gives more weight to classes that appear less frequently.

The purpose is to reduce the influence of class imbalance during training.

The same weighting is applied to the candidate models that support `sample_weight`.

Models that do not support the supplied training interface are skipped and logged rather than stopping the entire pipeline.

---

# Model Selection

Models are compared using the validation set.

The primary selection metric is:

```text
Macro F1
```

The validation results include:

* Model name
* Training time
* Validation accuracy
* Validation macro F1

The results are sorted by validation macro F1.

The highest validation macro-F1 model is selected as the candidate model.

---

# Why Macro F1?

This is a three-class classification problem:

```text
Bear
Neutral
Bull
```

Accuracy alone can hide poor performance on an individual class, particularly when the classes are imbalanced.

Macro F1 calculates F1 for each class and then gives each class equal importance.

Conceptually:

```text
Macro F1 =
(F1_Bear + F1_Neutral + F1_Bull) / 3
```

This makes it useful for evaluating performance across all three directional classes.

---

# Test Evaluation

The test set is deliberately kept separate from model selection.

The process is:

```text
Training set
     ↓
Train all models
     ↓
Validation set
     ↓
Select best model
     ↓
Test set
     ↓
Final evaluation
```

The test set is evaluated **once after model selection**.

This prevents the test set from gradually becoming another validation set.

Final metrics include:

```text
Test Accuracy
Test Macro F1
```

---

# MLflow Experiment Tracking

MLflow records the training experiments.

The configured experiment is:

```text
binance-ml-pipeline
```

The pipeline records information such as:

### Parameters

```text
model_name
symbol
horizon
val_ratio
test_ratio
```

### Metrics

```text
val_accuracy
val_f1_macro
train_time_s
test_accuracy
test_f1_macro
```

### Tags

The final model is tagged with:

```text
final_model=true
deployment_ready=true
```

This allows experiments and model versions to be inspected later rather than relying only on terminal output.

---

# MLflow Model Registry

The trained model is registered using a symbol-and-horizon-specific name.

Naming convention:

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

This keeps different prediction problems separate.

For example:

```text
BTCUSDT_30m_classifier
```

and

```text
BTCUSDT_4h_classifier
```

represent different learning problems and have independent model version histories.

---

# Deploy-if-Better Gate

The pipeline includes a deployment gate:

```python
deploy_if_better()
```

The logic is:

```text
New trained model
       │
       ▼
Register
       │
       ▼
Staging
       │
       ▼
Is there a Production model?
       │
      / \
    No   Yes
    │     │
    │     ▼
    │   Compare test F1
    │     │
    │     ▼
    │   Improvement >= 0.01?
    │      / \
    │    Yes  No
    │     │    │
    ▼     ▼    ▼
Production  Production  Stay Staging
```

If there is no existing Production model, the first registered model can become Production.

For subsequent models, the candidate must improve the incumbent's test macro F1 by at least:

```text
0.01
```

If it does not, it remains in Staging.

This creates a simple deployment safety gate instead of replacing the Production model after every training run.

---

# Model Artifact

The model logged to MLflow is not only the classifier.

The pipeline bundles:

```text
Preprocessor
     +
Classifier
     ↓
Single sklearn Pipeline
     ↓
MLflow Model Artifact
```

Conceptually:

```python
Pipeline([
    ("preprocessor", fitted_preprocessor),
    ("classifier", best_model)
])
```

This is important for deployment.

The serving API does not need to manually reconstruct the scaler, imputer, or classifier.

It loads the complete pipeline:

```text
Raw inference data
       ↓
MLflow model
       ↓
Preprocessing
       ↓
Classifier
       ↓
Prediction
```

The model artifact is stored using the `skops` format.

---

# Prefect Orchestration

### File

```text
flow.py
```

Prefect connects the individual tasks into one reproducible workflow.

The flow:

```python
@flow(name="binance-ml-pipeline")
```

contains:

```text
acquire_data()
       ↓
preprocess_data()
       ↓
engineer_features()
       ↓
align_and_save()
       ↓
train_and_evaluate()
       ↓
deploy_model()
```

Tasks also have retry and timeout configuration.

For example, data acquisition uses:

```text
retries = 3
retry delay = 10 seconds
timeout = 600 seconds
```

This is particularly useful for network-dependent operations such as downloading Binance data.

---

# Command-Line Interface

### File

```text
main.py
```

`main.py` provides a command-line interface around the Prefect flow.

### Basic execution

```bash
python main.py
```

This uses the default configuration:

```text
BTCUSDT
15m candles
30m prediction horizon
2023-01 → 2024-01
15% validation
15% test
0.002 target threshold
```

---

## Change the Trading Pair

```bash
python main.py --symbol ETHUSDT
```

---

## Change the Prediction Horizon

```bash
python main.py --horizon 1h
```

The horizon must be compatible with the selected candle interval.

---

## Change the Target Threshold

```bash
python main.py --target-threshold 0.003
```

This changes the threshold from:

```text
0.2%
```

to:

```text
0.3%
```

---

## Change the Date Range

```bash
python main.py \
  --start-year 2022 \
  --start-month 1 \
  --end-year 2024 \
  --end-month 1
```

---

## Change Validation/Test Ratios

```bash
python main.py \
  --val-ratio 0.20 \
  --test-ratio 0.10
```

This produces:

```text
70% Train
20% Validation
10% Test
```

---

# Output Data

The pipeline stores intermediate datasets under:

```text
data/
├── raw/
├── processed/
└── features/
```

### Raw

```text
data/raw/
└── BTCUSDT_15m_raw.parquet
```

Contains the downloaded and combined Binance market data.

### Processed

```text
data/processed/
└── cleaned_data.parquet
```

Contains cleaned data and target information.

### Features

```text
data/features/
├── X_final.parquet
└── y_final.parquet
```

`X_final.parquet` contains the final model features.

`y_final.parquet` contains the aligned target labels.

---

# Directory Structure

```text
3_Pipeline/
│
├── config/
│   ├── config.py
│   └── __init__.py
│
├── data/
│   ├── raw/
│   │   └── BTCUSDT_15m_raw.parquet
│   │
│   ├── processed/
│   │   ├── cleaned_data.parquet
│   │   └── cleaned_dummy_path.parquet
│   │
│   └── features/
│       ├── X_final.parquet
│       └── y_final.parquet
│
├── src/
│   ├── Data/
│   │   ├── data_acquisition.py
│   │   └── data_preprocessing.py
│   │
│   ├── Features/
│   │   └── feature_engineering.py
│   │
│   ├── Models/
│   │   ├── model_training.py
│   │   └── model_registry.py
│   │
│   └── Utils/
│       ├── logging_utils.py
│       └── retry_utils.py
│
├── flow.py
├── main.py
│
├── mlflow_binance.db
├── mlartifacts/
├── models/
├── logs/
├── catboost_info/
│
└── README.md
```

---

# Role of Each Source File

| File                     | Responsibility                                                   |
| ------------------------ | ---------------------------------------------------------------- |
| `flow.py`                | Prefect orchestration                                            |
| `main.py`                | Command-line entry point                                         |
| `config/config.py`       | Central configuration                                            |
| `data_acquisition.py`    | Downloads and prepares Binance Vision data                       |
| `data_preprocessing.py`  | Validates, cleans, and creates targets                           |
| `feature_engineering.py` | Calculates technical features                                    |
| `model_training.py`      | Splits data, preprocesses, trains, evaluates, and selects models |
| `model_registry.py`      | Handles MLflow registration and model stage transitions          |
| `logging_utils.py`       | Logging utilities                                                |
| `retry_utils.py`         | Retry-related utilities                                          |

---

# Complete Data Flow

The complete pipeline can be summarized as:

```text
                    ┌─────────────────────┐
                    │   Binance Vision    │
                    │   Historical Data   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Data Acquisition   │
                    │  Download + Combine  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Data Preprocessing  │
                    │ Validate + Clean    │
                    │ Create Target       │
                    └──────────┬──────────┘
                               │
                         X + y │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Engineering │
                    │ Returns / EMA / RSI │
                    │ Volatility / Volume │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Align X and y       │
                    │ Remove warm-up NaNs │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Chronological Split │
                    │ 70 / 15 / 15        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Preprocessing       │
                    │ Imputer + Scaler    │
                    └──────────┬──────────┘
                               │
                               ▼
          ┌─────────────────────────────────────────┐
          │             Model Benchmark             │
          │                                         │
          │ LR | RF | ET | XGB | LGBM | CatBoost   │
          │ GB | AdaBoost | SVC | KNN               │
          └──────────────────────┬──────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────┐
                    │ Validation F1       │
                    │ Model Selection     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Final Test          │
                    │ Accuracy + Macro F1 │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ MLflow              │
                    │ Tracking + Registry │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Deployment Gate     │
                    │ Staging / Production│
                    └─────────────────────┘
```

---

# Reproducibility and Leakage Prevention

Several design decisions specifically address data leakage.

### 1. Chronological splitting

Data is never randomly shuffled before train/validation/test splitting.

### 2. Training-only preprocessing fit

The imputer and scaler are fitted using the training set only.

### 3. Forward-fill missing values

Missing market values are filled using previous observations rather than future values.

### 4. Target created independently

The target is calculated from future prices and is kept separate from feature engineering.

### 5. Future target columns are removed from X

The following columns never enter the model:

```text
future_close
future_return
target
```

### 6. Test set used only after model selection

The test set is evaluated only after the validation set has selected the final candidate.

These controls are particularly important for financial time-series problems because seemingly small leakage errors can produce misleadingly optimistic results.

---

# MLflow Storage

The training pipeline uses its own MLflow tracking environment.

The local training artifacts include:

```text
mlflow_binance.db
mlartifacts/
```

Conceptually:

```text
MLflow Database
      │
      ├── Experiments
      ├── Runs
      ├── Parameters
      ├── Metrics
      └── Model Registry metadata

MLflow Artifacts
      │
      └── Trained model files
```

The deployment environment may use a **separate MLflow instance/database**.

The trained model therefore does not automatically appear in the deployment registry simply because it was trained here.

A controlled migration/registration step is used when moving a selected model into the deployment environment.

---

# Relationship With Online Deployment

This pipeline is responsible for producing the model.

The online deployment is responsible for serving it.

```text
3_Pipeline
──────────
Historical data
      ↓
Training
      ↓
Evaluation
      ↓
MLflow
      ↓
Registered model
      │
      │ migration / deployment
      ▼
4_Deploy_Online
───────────────
MLflow Production model
      ↓
FastAPI
      ↓
Live prediction
```

The online API does not retrain models.

It loads the model that has been selected for Production and applies it to newly retrieved market data.

---

# Current Example

For the current MVP, the pipeline produced a model registered as:

```text
BTCUSDT_30m_classifier
```

The model selected for the current 30-minute workflow is an AdaBoost classifier.

The training and evaluation metrics are tracked in MLflow rather than being treated as hardcoded values in the deployment code.

The resulting registered model can then be moved into the separate deployment environment and promoted to Production.

---

# Future Pipeline Improvements

The pipeline architecture is designed to grow with the project.

Potential improvements include:

### Data

* Additional cryptocurrencies
* Additional trading pairs
* More historical periods
* Additional market data sources
* Automated incremental data updates

### Features

* MACD
* ATR / NATR
* Bollinger Bands
* EMA-200 relationships
* Additional RSI periods
* More volume indicators
* Market-wide features
* Order-book features

### Modeling

* Hyperparameter optimization
* Feature selection
* Probability calibration
* Ensemble optimization
* Time-series cross-validation
* More specialized temporal models

### MLOps

* Scheduled retraining
* Automated data-quality checks
* Data drift monitoring
* Model drift monitoring
* Automated experiment comparison
* CI/CD integration
* Model rollback
* Improved registry alias management

---

# Important Notes

### This is a prediction system, not a trading bot

The pipeline predicts:

```text
Bear / Neutral / Bull
```

It does not execute trades.

### Predictions are not guarantees

Financial markets are noisy and affected by many factors that are not represented in the current feature set.

Model performance on historical data therefore does not guarantee future performance.

### MVP scope is intentionally limited

The current implementation focuses on:

```text
BTCUSDT
+
15-minute candles
+
30-minute prediction horizon
```

The architecture supports expansion to additional symbols and horizons as the project develops.

---

# Running the Pipeline

From this directory:

```bash
cd 3_Pipeline
```

Run with defaults:

```bash
python main.py
```

Run with custom configuration:

```bash
python main.py \
  --symbol BTCUSDT \
  --interval 15m \
  --start-year 2023 \
  --start-month 1 \
  --end-year 2024 \
  --end-month 1 \
  --target-threshold 0.002 \
  --horizon 30m \
  --val-ratio 0.15 \
  --test-ratio 0.15
```

The Prefect flow can also be invoked directly through:

```bash
python flow.py
```

---

# Pipeline Summary

The pipeline follows a clear MLOps lifecycle:

```text
ACQUIRE
   ↓
Historical Binance data

PREPROCESS
   ↓
Clean data + target

FEATURE ENGINEER
   ↓
Market features

SPLIT
   ↓
Chronological train / validation / test

TRAIN
   ↓
10 candidate classifiers

SELECT
   ↓
Best validation Macro F1

TEST
   ↓
Final unseen evaluation

TRACK
   ↓
MLflow experiment

REGISTER
   ↓
Versioned model

DEPLOY GATE
   ↓
Staging → Production when criteria are met
```

The result is a **versioned, reproducible, deployment-ready machine-learning model** that can be consumed by the project's online inference service.
