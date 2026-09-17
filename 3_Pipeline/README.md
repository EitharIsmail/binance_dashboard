# Binance ML Pipeline — Pipeline & Orchestration Reference

This document explains what each file in the pipeline does, why it exists, and exactly what it returns to the next stage. Read it top-to-bottom to follow data as it flows from a raw Binance download through to a deployed, servable model.

**Pipeline order:** `data_acquisition.py` → `data_preprocessing.py` → `feature_engineering.py` → `model_training.py` → `model_registry.py` → `model_deployment.py`, all orchestrated by `flow.py` and launched from `main.py`.

---

## 1. `data_acquisition.py` — `BinanceDataAcquisition`

### Where the data comes from, and why

Data is pulled from **Binance Vision** (`https://data.binance.vision`), Binance's public historical-data archive. Binance publishes monthly ZIP files of raw OHLCV (open/high/low/close/volume) candlestick data per symbol/interval combination — e.g. one file per month for `BTCUSDT` at `15m` candles. This is used instead of Binance's live REST/WebSocket API because:

- It's free, requires no API key, and has no rate-limiting concerns for historical backfills.
- It's the *canonical* historical record — the same source Binance's own charts are built from — so results are reproducible.
- Pulling a year+ of 15-minute candles via the live API would mean thousands of paginated calls; the monthly archive gets the same data in a handful of large downloads.

### What it does

1. `generate_monthly_urls()` builds one Binance Vision URL per calendar month in the requested date range (using month-start (`'MS'`) date stepping to avoid a 30-day drift bug that occurs with naive day-counting).
2. `download_and_extract_zip()` downloads each monthly ZIP, extracts the single CSV inside it, and assigns Binance's fixed 12-column kline schema (`open_time`, `open`, `high`, `low`, `close`, `volume`, `close_time`, `quote_asset_volume`, `number_of_trades`, `taker_buy_base_asset_volume`, `taker_buy_quote_asset_volume`, `ignore`).
3. `process_timestamps_and_types()` fixes a known Binance quirk where some months ship timestamps in microseconds instead of milliseconds (detected via magnitude, `> 1e14`), and casts OHLCV columns to `float`.
4. `combine_and_clean_dataframes()` concatenates every month, sorts chronologically by `open_time`, and drops any duplicate timestamps (can occur at month boundaries).
5. Saves the combined result to `data/raw/{symbol}_{interval}_raw.parquet` for reproducibility, so the exact same raw pull can be reloaded without re-downloading.

### What the file returns

`BinanceDataAcquisition.run(start_date, end_date)` returns a single **raw** `pandas.DataFrame`:

- One row per candle, sorted chronologically, no duplicate timestamps
- 12 columns: the untouched Binance kline schema listed above
- `open_time` as a real `datetime64` column; OHLCV columns as `float`
- **Nothing engineered yet** — no target, no indicators. This is the rawest usable form of the market data.

---

## 2. `data_preprocessing.py` — `DataPreprocessor`

### What we got, and what we did with it

This stage takes the raw OHLCV DataFrame from Step 1 and does two distinct jobs: **cleaning** the data, and **creating the prediction target** (`y`). Target creation lives here — not in feature engineering — because the label only ever depends on the raw `close` price; it needs no engineered indicator to be computed.

**Cleaning steps** (`validate_and_cast_types`, `sort_and_deduplicate`, `handle_missing_values`):
- Re-validates/casts `open_time` to datetime and core OHLCV columns to numeric, defensively, in case the raw parquet was reloaded from disk with lost dtype info.
- Re-sorts chronologically and re-drops duplicate timestamps (defensive — should already be clean from Step 1, but this stage doesn't assume that).
- Fills missing values via **forward-fill only** (never backward-fill or interpolation) — this is deliberate: backward-filling or interpolating would let a *future* value leak into a *past* row, which is a subtle form of data leakage in a time-series model. Any row still missing core data after forward-fill is dropped outright.

**Target creation** (`create_target`, `drop_unlabelable_rows`):
- For every row, looks `horizon_periods` candles into the future (`horizon` ÷ `interval` — e.g. `30m` horizon at `15m` candles = 2 periods ahead) and computes the forward return: `(future_close − close) / close`.
- Labels the row:
  - **`1` (Bull)** if the forward return ≥ `target_threshold` (default 0.2%)
  - **`-1` (Bear)** if the forward return ≤ `-target_threshold`
  - **`0` (Neutral)** otherwise
- The last `horizon_periods` rows of the dataset have no future candle to look at (the shift runs off the end of the series) — these are unlabelable and are dropped.

### Why we calculate `y` this way

The goal isn't to predict an exact future price (regression) — it's to predict *direction with a meaningful magnitude*, filtered through a noise threshold. A pure up/down (2-class) label would treat tiny, directionless wiggles the same as a real move, which is mostly noise at short horizons. The 3-class Bull/Neutral/Bear scheme with a threshold explicitly separates "this is a real, tradeable move" from "this is noise around zero" — which is also why the resulting classes are imbalanced (Neutral dominates at tight thresholds/short horizons), a fact that later stages (`compute_sample_weight("balanced", ...)` in model training) explicitly correct for.

### What the file returns

`DataPreprocessor.run(df)` returns a tuple **`(X, y)`**:

- **`X`**: a DataFrame containing every original raw column *except* the three target-derived helper columns (`future_close`, `future_return`, `target` itself). Still includes `open_time` and unmodified OHLCV — no technical indicators yet.
- **`y`**: an integer `pandas.Series`, one label per remaining row, valued `{-1, 0, 1}` for Bear/Neutral/Bull.
- Also persists the full cleaned+labeled DataFrame to `data/processed/cleaned_{input_stem}.parquet` for inspection/reproducibility.

---

## 3. `feature_engineering.py` — `FeatureEngineer`

### What it does

Takes `X` from Step 2 (raw OHLCV, no indicators) and computes technical indicators — the actual predictive signals the models will learn from. Implemented as a scikit-learn-compatible `TransformerMixin`, so it can later be slotted into a full sklearn `Pipeline` alongside the preprocessor and classifier.

### Engineered features

| Feature | Formula / Definition | How it contributes to predicting `y` |
|---|---|---|
| `return_1` | 1-period percent change in `close` | The most recent single-candle momentum — direct, short-term directional signal. |
| `return_3` | 3-period percent change in `close` | A slightly smoothed momentum reading; filters single-candle noise that `return_1` alone can't. |
| `volatility_10` | Rolling 10-period standard deviation of `return_1` | Measures how turbulent the market currently is. High volatility makes the Bull/Bear threshold easier to cross (more relevant when combined with the target's fixed % threshold), so this helps the model gauge *how likely* a real move is right now, independent of direction. |
| `ema_20` | Exponential moving average, span 20 | A smoothed short-term trend line — where price is "centered" over the recent past. |
| `ema_50` | Exponential moving average, span 50 | A smoothed medium-term trend line. Comparing `ema_20` vs `ema_50` (implicitly, via the two separate columns) lets tree-based models learn crossover-style trend signals. |
| `dist_from_ema_20` | `(close − ema_20) / ema_20` | How far current price has stretched away from its short-term trend, as a percentage. Large deviations often precede mean-reversion (pulling back toward Neutral) or trend continuation — a key input for distinguishing a real breakout from a temporary spike. |
| `rsi_14` | Relative Strength Index, 14-period (Wilder's smoothing via `ewm(alpha=1/14)`) | Classic overbought/oversold oscillator (0–100 scale). Extreme values (near 0 or 100) are associated with exhaustion of a move — directly relevant to whether a Bull/Bear move is likely to continue or reverse to Neutral. |
| `volume_sma_20` | Rolling 20-period simple average of `volume` | Baseline "typical" trading volume, used only as the denominator for `volume_ratio` below. |
| `volume_ratio` | `volume / volume_sma_20` | Current volume relative to its recent normal level. A move on unusually high volume is more likely to be a genuine, sustained directional move (Bull/Bear) than one on thin volume, which is more likely noise (Neutral). |

### Why NaNs appear here (and not earlier)

Every rolling/EWM feature above needs a warm-up window before it has enough history to compute a real value (e.g. `ema_50` needs ~50 prior rows). These NaNs don't exist until this transform runs — which is exactly why `align_features_and_target()` (also in this file) has to run *after* feature engineering, not inside `DataPreprocessor`.

### What the file returns

- `FeatureEngineer.fit_transform(X)` returns `X` with all 9 columns above appended (original columns preserved) — shape grows from the ~12 raw columns to 21.
- `align_features_and_target(X, y)` returns a tuple **`(X_aligned, y_aligned)`**: both re-indexed to drop the warm-up NaN rows and stay row-aligned with each other, ready for the train/val/test split.

---

## 4. `model_training.py` — model bench, preprocessing, and `ModelTrainingPipeline`

### What it does, briefly

This is the largest stage: it splits the fully-featured data chronologically into train/val/test, fits a numeric-only imputer+scaler, benches 10 classifiers against the validation set, and evaluates the winner once on the held-out test set — all wrapped in one `ModelTrainingPipeline.run(X, y)` call. `LABEL_MAP`/`INVERSE_LABEL_MAP` remap the `{-1, 0, 1}` Bear/Neutral/Bull labels to `{0, 1, 2}`, since XGBoost/LightGBM/CatBoost require contiguous non-negative class labels — this mapping is the single source of truth other files (`model_deployment.py`) rely on to translate predictions back to human-readable labels.

### The 10 candidate models

| Model | Key hyperparameters set | What they control | Fixed/housekeeping args |
|---|---|---|---|
| Logistic Regression | (none tuned — defaults used) | Linear baseline; `max_iter=1000` just ensures convergence, not a real hyperparameter | `random_state=42` (reproducibility) |
| Random Forest | `n_estimators=200` (# trees), `max_depth=10` (tree depth cap) | Bias/variance tradeoff: more trees reduce variance via averaging; capped depth prevents individual trees from memorizing noise | `n_jobs=-1` (use all CPU cores), `random_state=42` |
| Extra Trees | Same as RF: `n_estimators=200`, `max_depth=10` | Like RF, but splits are chosen randomly rather than by best-split search — trades a bit of per-tree accuracy for much lower correlation between trees, often reducing variance further | same |
| XGBoost | `n_estimators=200`, `max_depth=6`, `learning_rate=0.05` | Boosting-specific: shallower trees than RF (boosting builds complexity additively), and a conservative learning rate paired with 200 rounds | `objective="multi:softprob"` (3-class probability output), `eval_metric="mlogloss"` (internal loss tracking) |
| LightGBM | Same trio as XGBoost | Same tradeoffs, different tree-growth algorithm (leaf-wise vs level-wise — LightGBM tends to fit faster on large data) | `objective="multiclass"` |
| CatBoost | `iterations=200` (=n_estimators), `depth=6`, `learning_rate=0.05` | Same boosting logic; CatBoost's distinguishing feature (ordered boosting, native categorical handling) isn't relevant here since all your features are numeric | `loss_function="MultiClass"` |
| Gradient Boosting (sklearn) | `n_estimators=100`, `max_depth=5`, `learning_rate=0.05` | sklearn's native (slower) boosting implementation — same concept as XGBoost, fewer rounds since it's the most compute-expensive of the boosters (71s vs XGBoost's 5s observed in Step 10) | — |
| AdaBoost | `n_estimators=100`, `learning_rate=0.05` | Reweights misclassified samples each round rather than fitting residuals — a different, older boosting mechanism. No `max_depth` because it boosts decision stumps by default (depth-1 trees) | — |
| SVC (RBF) | (kernel fixed, no C/gamma tuned) | `kernel="rbf"` picks a nonlinear decision boundary; `C` and `gamma` — the two hyperparameters that actually matter most for SVM performance — are left at sklearn defaults (`C=1.0`, `gamma="scale"`), meaning this model is essentially untuned out of the box | `probability=True` (enables `.predict_proba()`, deprecated in recent sklearn — costly, since it fits an internal calibration model on top) |
| KNN | `n_neighbors=15` | Only real hyperparameter for KNN; 15 is a reasonable default but arbitrary | `n_jobs=-1` |

**Note:** as shipped, none of the 10 models are hyperparameter-tuned — this table describes hand-picked defaults, not search results. `train_all_models()` fits every model with **balanced sample weights** (`compute_sample_weight("balanced", ...)`) to correct for the Neutral-class imbalance inherited from Step 2; KNN is the one exception, since `KNeighborsClassifier.fit()` doesn't accept `sample_weight` and is skipped with a logged warning rather than crashing the run.

### Preprocessing and split logic

- `chronological_train_val_test_split()` splits by row position (never shuffled — shuffling would leak adjacent rolling-window information across the boundary), producing train → val → test in strict time order, so the test set represents the most recent, truly unseen period.
- `fit_preprocessor()` builds a `ColumnTransformer` (median imputer → `StandardScaler`) fit only on `X_train`'s **numeric columns** — non-numeric columns like `open_time` are automatically excluded, so no manual column-dropping is needed.

### What the file returns

`ModelTrainingPipeline.run(X, y)` returns a dict:

- `best_model_name` — the model with the highest validation macro F1
- `best_model` — the fitted estimator object itself
- `preprocessor` — the fitted `ColumnTransformer`
- `val_results` — the full 10-model comparison table (sorted best-first)
- `test_metrics` — `{"test_accuracy": ..., "test_f1_macro": ...}` from the **one-time** held-out test evaluation
- `run_id` — the MLflow run ID everything above was logged under (params + metrics), which `model_registry.py` uses to register the model

---

## 5. `model_registry.py` — `ModelRegistry`

### What it does

Wraps MLflow's Model Registry client operations into one class, so the rest of the pipeline doesn't have to talk to `MlflowClient` directly. It manages the lifecycle of a **named, versioned model** through MLflow's stage system:

- `register_model(run_id)` — takes a model already logged inside an MLflow run (via `mlflow.sklearn.log_model(...)`) and registers it as a new numbered version under `self.model_name`.
- `transition_to_staging(run_id, description, tags)` — moves a version into the `Staging` stage, attaching a human-readable description and optional metadata tags (e.g. validation scores).
- `transition_to_production(version)` — promotes a version to `Production`, and **automatically archives whatever was previously in `Production`** first, so there's only ever one active production version at a time.
- `get_model_by_stage(stage)` / `get_all_versions()` — read-only lookups used both by `print_registry_status()` (a human-readable console summary) and by `model_deployment.py` to find the currently-deployed version.
- `get_deployment_uri(stage)` — builds the `models:/{name}/{stage}` URI string that MLflow's `load_model()` functions expect.

**Note on the uploaded `model_deployment.py`:** it imports `load_best_model_pointer`, `load_preprocessor`, and `load_model` from `model_registry` — none of which exist in the `model_registry.py` shown here (that file only defines the `MLflowConfig` dataclass and `ModelRegistry` class). This suggests either an earlier, file/joblib-based registry implementation (a "pointer" JSON file naming the current best model, plus separately joblib-dumped preprocessor/model files) that predates the MLflow-based version, or a registry module still to be written. These two files as currently uploaded are **not compatible with each other** — resolving that mismatch (either by adding the three missing functions to `model_registry.py`, or rewriting `model_deployment.py` to use `ModelRegistry`'s MLflow-based methods instead) is a prerequisite before deployment will actually run.

### What the file returns

`ModelRegistry`'s methods don't return a single "pipeline output" the way the earlier stages do — each method returns what its specific registry action produces:

- `register_model()` → the new version number (`str`) or `None` on failure
- `transition_to_staging()` → the version number that was transitioned, or `None`
- `transition_to_production()` → `True`/`False` success flag
- `get_model_by_stage()` → a single `ModelVersion` object or `None`
- `get_all_versions()` → a list of `ModelVersion` objects, sorted oldest→newest
- `get_deployment_uri()` → a `models:/{name}/{stage}` string, ready to hand to `mlflow.sklearn.load_model()`

---

## 6. `model_deployment.py` — `Predictor`

### What it does

The inference-time entry point — where a trained, registered model actually gets used to make predictions on new data. `Predictor.__init__` loads three things needed for inference (currently via the not-yet-defined registry functions flagged above): a "pointer" describing which model is currently marked best, the fitted preprocessor, and the model itself.

`predict(X_new)`:
1. Runs `X_new` (raw, unscaled feature rows — same schema as what `FeatureEngineer` + alignment produce) through the loaded preprocessor.
2. Runs the transformed features through the model, getting predictions in the `{0, 1, 2}` training label space.
3. Maps predictions back to the original, human-meaningful `{-1, 0, 1}` (Bear/Neutral/Bull) space via `INVERSE_LABEL_MAP` from `model_training.py` — callers never see the internal `{0,1,2}` encoding.

`predict_proba(X_new)` does the same but returns class probabilities instead of hard labels, for any downstream use that wants confidence scores rather than a single decision — with an explicit check that raises if the loaded model doesn't support `predict_proba` (not every model in the bench does).

### What the file returns

- `Predictor.predict(X_new)` → a NumPy array of `{-1, 0, 1}` labels, one per input row
- `Predictor.predict_proba(X_new)` → a NumPy array of shape `(n_rows, 3)`, class probabilities in `{Bear, Neutral, Bull}` training order
- The module-level `predict(X_new, model_dir)` convenience function → same as `Predictor.predict()`, for one-off calls where a caller doesn't need to keep the loaded model in memory

---

## 7. `flow.py` — Prefect orchestration

### What it does

Defines the end-to-end pipeline as a Prefect `@flow`, composed of five `@task`-decorated stages that wrap the classes/functions described above:

1. **`acquire-data`** (`retries=3`, `retry_delay_seconds=10`, `timeout_seconds=600`) — wraps `BinanceDataAcquisition.run()`. Generous retry budget because this is the one stage doing real network I/O against an external host.
2. **`preprocess-data`** (`retries=1`) — wraps `DataPreprocessor.run()`, called with the in-memory DataFrame from Step 1 rather than re-reading from disk.
3. **`engineer-features`** (`retries=1`) — wraps `FeatureEngineer.fit_transform()`.
4. **`align-and-save`** (`retries=1`) — wraps `align_features_and_target()`, then persists the final `X`/`y` to parquet **and** returns the in-memory DataFrames directly to the next task (so the training stage doesn't need a redundant disk round-trip).
5. **`train-and-evaluate`** (`retries=1`) — wraps `ModelTrainingPipeline.run()`, with `tracking_uri` pulled from `config.mlflow.tracking_uri` at module load time.

Dependencies between tasks aren't declared explicitly — Prefect infers them from how return values of one task are passed as arguments into the next (e.g. `preprocess_data(raw_df=df_raw, ...)` implies `preprocess-data` can't start until `acquire-data` finishes). The `@flow`-decorated `binance_ml_pipeline()` function is the orchestrator that calls all five tasks in sequence and logs a final summary.

### What the file returns

`binance_ml_pipeline(...)` returns a 3-tuple: **`(x_path, y_path, training_result)`** — the on-disk paths to the final feature/target parquet files, plus the full result dict from `ModelTrainingPipeline.run()` (best model, its metrics, the MLflow run ID, etc).

---

## 8. `main.py` — CLI entry point

### What it does

The script users actually run (`python main.py [options]`). It:

1. Configures root Python logging (`configure_logging()`) so that plain `logging.getLogger(__name__)` calls inside `DataPreprocessor`/`FeatureEngineer` (which don't use Prefect's `get_run_logger()`) still print to the console — Prefect's own task-level logging works independently of this.
2. Parses CLI arguments (`parse_args()`) covering every configurable aspect of a run: symbol, interval, date range, output directories, target threshold, horizon, val/test split ratios, and log level.
3. Calls `binance_ml_pipeline(...)`, passing through every parsed argument.
4. Logs a final human-readable summary: output paths, the winning model's name, its test accuracy/F1, and the MLflow run ID it was logged under — everything needed to go find that run in the MLflow UI afterward.

### What the file returns

`main()` doesn't return a value (it's a script entry point) — its output is entirely side-effecting: console log lines, the parquet files written by the flow, and the MLflow run created by `ModelTrainingPipeline`. Running `python main.py` end-to-end is the trigger for the entire chain described above, from a fresh Binance download through to a registered, evaluated model.