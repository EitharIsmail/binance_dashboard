import time
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import mlflow

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from config.config import MLflowConfig

config = MLflowConfig()

logger = logging.getLogger(__name__)

# Target labels are -1 (Bear), 0 (Neutral), 1 (Bull). XGBoost/LightGBM/CatBoost
# require contiguous non-negative class labels. registry.py and deployment.py
# both import these two maps so persistence and inference agree on them.
LABEL_MAP = {-1: 0, 0: 1, 1: 2}
INVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def get_default_models() -> Dict[str, object]:
    """The 10-model bench from the notebook."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            objective="multi:softprob", eval_metric="mlogloss",
            random_state=42, n_jobs=-1, verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            objective="multiclass", random_state=42, n_jobs=-1, verbose=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=200, depth=6, learning_rate=0.05,
            loss_function="MultiClass", random_state=42, verbose=0,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100, learning_rate=0.05, random_state=42
        ),
        "SVC (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=15, n_jobs=-1),
    }


def build_preprocessor(feature_names: list) -> ColumnTransformer:
    """Imputer + StandardScaler. Caller is responsible for fitting on
    training data only -- this function just constructs the (unfitted)
    transformer."""
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), feature_names)
        ],
        remainder="drop",
    )


def chronological_train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Row-index split into train/val/test (data must already be sorted
    chronologically upstream). No shuffling -- shuffling would leak
    adjacent rolling-window information across split boundaries.

    Order is preserved in time: train comes first, then val, then test,
    so the test set represents the most recent, truly unseen period.
    """
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    n = len(X)
    train_end = int(n * (1 - val_ratio - test_ratio))
    val_end = int(n * (1 - test_ratio))

    X_train = X.iloc[:train_end].reset_index(drop=True)
    X_val = X.iloc[train_end:val_end].reset_index(drop=True)
    X_test = X.iloc[val_end:].reset_index(drop=True)

    y_train = y.iloc[:train_end].reset_index(drop=True)
    y_val = y.iloc[train_end:val_end].reset_index(drop=True)
    y_test = y.iloc[val_end:].reset_index(drop=True)

    logger.info(
        f"Chronological split -> train: {len(X_train):,} rows, "
        f"val: {len(X_val):,} rows, test: {len(X_test):,} rows"
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def fit_preprocessor(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: Optional[pd.DataFrame] = None,
):
    """Fits the preprocessor on X_train, transforms train/val (and test,
    if provided). Only numeric columns are used as features -- non-numeric
    columns (e.g. `open_time`, a datetime) are excluded automatically.
    Returns (preprocessor, X_train_proc, X_val_proc) or
    (preprocessor, X_train_proc, X_val_proc, X_test_proc) if X_test is given."""
    numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
    preprocessor = build_preprocessor(numeric_cols)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)

    if X_test is not None:
        X_test_proc = preprocessor.transform(X_test)
        return preprocessor, X_train_proc, X_val_proc, X_test_proc

    return preprocessor, X_train_proc, X_val_proc


def train_all_models(
    X_train_proc,
    y_train: pd.Series,
    models: Optional[Dict[str, object]] = None,
) -> Dict[str, dict]:
    """
    Fits every model in `models` with balanced sample weights.

    Returns {name: {"model": fitted_model, "train_time": seconds}}.
    Models that fail to fit (e.g. KNN, which doesn't accept sample_weight)
    are skipped with a logged warning rather than raising -- this matches
    the notebook's actual observed behaviour.

    Nothing is persisted here; that's registry.py's job.
    """
    if models is None:
        models = get_default_models()

    y_train_mapped = y_train.map(LABEL_MAP)
    sample_weights = compute_sample_weight("balanced", y_train_mapped)
    logger.info("Calculated balanced sample weights for training set.")

    fitted = {}
    for name, model in models.items():
        logger.info(f"Training {name}...")
        try:
            start_time = time.time()
            model.fit(X_train_proc, y_train_mapped, sample_weight=sample_weights)
            train_time = time.time() - start_time
            fitted[name] = {"model": model, "train_time": train_time}
            logger.info(f"  {name} trained in {train_time:.2f}s")
        except Exception as e:
            logger.warning(f"  {name} failed to train: {e}")

    return fitted


def evaluate_models(
    fitted_models: Dict[str, dict],
    X_val_proc,
    y_val: pd.Series,
) -> pd.DataFrame:
    """
    Scores every fitted model against the validation set.
    Returns a results DataFrame sorted by validation macro F1 (best first).
    Nothing is persisted here.
    """
    y_val_mapped = y_val.map(LABEL_MAP)

    results = []
    for name, entry in fitted_models.items():
        model = entry["model"]
        y_val_pred = model.predict(X_val_proc)
        acc = accuracy_score(y_val_mapped, y_val_pred)
        f1_macro = f1_score(y_val_mapped, y_val_pred, average="macro")
        results.append({
            "Model": name,
            "Train Time (s)": round(entry["train_time"], 2),
            "Validation Accuracy": round(acc, 4),
            "Validation Macro F1": round(f1_macro, 4),
        })

    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results = df_results.sort_values(
            by="Validation Macro F1", ascending=False
        ).reset_index(drop=True)
    else:
        logger.warning("No models were successfully trained.")

    return df_results


def evaluate_on_test(model, X_test_proc, y_test: pd.Series, run_id: str) -> Dict[str, float]:
    """
    Evaluate a single, already-chosen model on the held-out test set and log
    the result to an existing MLflow run.

    Call this ONCE, after model selection is finalized on the validation set
    (evaluate_models() above) -- repeatedly checking test performance while
    still choosing between models turns the test set into a second
    validation set and defeats its purpose as an unbiased final check.

    `run_id` must already exist (e.g. from an mlflow.start_run() call made
    when the corresponding model was trained) -- this function does not
    create a new run, only logs into one that already exists.
    """
    logger.info("🔬 Evaluating on test set...")

    y_test_mapped = y_test.map(LABEL_MAP)
    y_test_pred = model.predict(X_test_proc)

    test_metrics = {
        "test_accuracy": accuracy_score(y_test_mapped, y_test_pred),
        "test_f1_macro": f1_score(y_test_mapped, y_test_pred, average="macro"),
    }

    logger.info(f"   Test Accuracy: {test_metrics['test_accuracy']:.4f}")
    logger.info(f"   Test Macro F1: {test_metrics['test_f1_macro']:.4f}")

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(test_metrics)
        mlflow.set_tag("final_model", "true")
        mlflow.set_tag("deployment_ready", "true")

    return test_metrics


# =============================================================================
# Optuna hyperparameter tuning — top-3 candidates only
# =============================================================================
def _tree_ensemble_params(trial):
    """Shared search space for Random Forest / Extra Trees."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
        "max_depth": trial.suggest_categorical("max_depth", [5, 10, 15, 20, None]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    }


def _boosting_params(trial):
    """Shared search space for XGBoost / LightGBM."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }


def _catboost_params(trial):
    return {
        "iterations": trial.suggest_int("iterations", 100, 400, step=50),
        "depth": trial.suggest_int("depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
    }


def _gb_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=25),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
    }


def _adaboost_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=25),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0, log=True),
    }


def _logreg_params(trial):
    return {"C": trial.suggest_float("C", 1e-3, 10.0, log=True), "max_iter": 1000}


# Maps each tunable model name -> (param-suggestion fn, model-builder fn).
# Builders re-attach the fixed/housekeeping args from get_default_models()
# (random_state, objective, n_jobs, etc.) around the tuned params.
TUNING_REGISTRY = {
    "Random Forest": (_tree_ensemble_params, lambda p: RandomForestClassifier(random_state=42, n_jobs=-1, **p)),
    "Extra Trees": (_tree_ensemble_params, lambda p: ExtraTreesClassifier(random_state=42, n_jobs=-1, **p)),
    "XGBoost": (_boosting_params, lambda p: XGBClassifier(
        objective="multi:softprob", eval_metric="mlogloss", random_state=42, n_jobs=-1, verbosity=0, **p)),
    "LightGBM": (_boosting_params, lambda p: LGBMClassifier(
        objective="multiclass", random_state=42, n_jobs=-1, verbose=-1, **p)),
    "CatBoost": (_catboost_params, lambda p: CatBoostClassifier(
        loss_function="MultiClass", random_state=42, verbose=0, **p)),
    "Gradient Boosting": (_gb_params, lambda p: GradientBoostingClassifier(random_state=42, **p)),
    "AdaBoost": (_adaboost_params, lambda p: AdaBoostClassifier(random_state=42, **p)),
    "Logistic Regression": (_logreg_params, lambda p: LogisticRegression(random_state=42, n_jobs=-1, **p)),
    # SVC and KNN intentionally omitted: SVC tuning (C, gamma) is expensive
    # at this dataset size, and KNN's only real hyperparameter (n_neighbors)
    # is cheap enough to leave at its fixed default.
}


def tune_model(
    name: str,
    X_train_proc,
    y_train_mapped: pd.Series,
    n_splits: int = 3,
    n_trials: int = 30,
) -> Tuple[object, float]:
    """
    Runs Optuna TPE search for one model using chronological
    (TimeSeriesSplit) CV, with per-fold balanced sample weights. Returns
    (refit_model, best_cv_f1). Raises KeyError if `name` has no entry in
    TUNING_REGISTRY.
    """
    suggest_fn, build_fn = TUNING_REGISTRY[name]
    tscv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial):
        params = suggest_fn(trial)
        fold_scores = []
        for fold_i, (tr_idx, va_idx) in enumerate(tscv.split(X_train_proc)):
            X_tr, X_va = X_train_proc[tr_idx], X_train_proc[va_idx]
            y_tr, y_va = y_train_mapped.iloc[tr_idx], y_train_mapped.iloc[va_idx]

            weights = compute_sample_weight("balanced", y_tr)
            model = build_fn(params)
            model.fit(X_tr, y_tr, sample_weight=weights)

            preds = model.predict(X_va)
            fold_scores.append(f1_score(y_va, preds, average="macro"))
            trial.report(np.mean(fold_scores), step=fold_i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    # Refit the winning config on the FULL training set (CV folds each
    # used less data than the full set).
    final_model = build_fn(study.best_params)
    full_weights = compute_sample_weight("balanced", y_train_mapped)
    final_model.fit(X_train_proc, y_train_mapped, sample_weight=full_weights)

    logger.info(f"  {name} tuned -> CV F1={study.best_value:.4f}, params={study.best_params}")
    return final_model, study.best_value



# =============================================================================
# Model Training Pipeline — wraps split -> preprocess -> train -> select -> test
# =============================================================================
class ModelTrainingPipeline:
    """
    End-to-end model selection pipeline:
      split -> preprocess -> train all candidates -> pick best on val ->
      log to MLflow (as a bundled preprocessor+classifier Pipeline,
      registered under a symbol/horizon-specific name) -> evaluate best
      model once on test.

    Call `.run(X, y, symbol=..., horizon=...)`. Intermediate state (fitted
    preprocessor, results table, chosen model) is kept on `self` for
    inspection after the run.
    """

    def __init__(
        self,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        models: Optional[Dict[str, object]] = None,
        experiment_name: str = "binance-ml-pipeline",
        tracking_uri: str = "http://127.0.0.1:5000",
    ):
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.models = models or get_default_models()
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri

        # populated by run()
        self.preprocessor = None
        self.val_results: Optional[pd.DataFrame] = None
        self.best_model_name: Optional[str] = None
        self.best_model = None
        self.run_id: Optional[str] = None
        self.test_metrics: Optional[dict] = None

    def run(self, X: pd.DataFrame, y: pd.Series, symbol: str, horizon: str) -> dict:
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        X_train, y_train, X_val, y_val, X_test, y_test = chronological_train_val_test_split(
            X, y, val_ratio=self.val_ratio, test_ratio=self.test_ratio
        )

        self.preprocessor, X_train_proc, X_val_proc, X_test_proc = fit_preprocessor(
            X_train, X_val, X_test
        )

        fitted_models = train_all_models(X_train_proc, y_train, models=self.models)
        self.val_results = evaluate_models(fitted_models, X_val_proc, y_val)

        if self.val_results.empty:
            raise RuntimeError("No models trained successfully — nothing to select from.")

        # --- Tune the top 3 baseline models (that have a search space defined) ---
        y_train_mapped = y_train.map(LABEL_MAP)
        y_val_mapped = y_val.map(LABEL_MAP)

        top_candidates = [
            name for name in self.val_results["Model"].head(3) if name in TUNING_REGISTRY
        ]
        logger.info(f"🎯 Tuning top {len(top_candidates)} candidates: {top_candidates}")

        tuned_rows = []
        for name in top_candidates:
            tuned_model, _ = tune_model(name, X_train_proc, y_train_mapped)
            preds = tuned_model.predict(X_val_proc)
            tuned_f1 = f1_score(y_val_mapped, preds, average="macro")
            tuned_acc = accuracy_score(y_val_mapped, preds)

            baseline_f1 = self.val_results.loc[
                self.val_results["Model"] == name, "Validation Macro F1"
            ].iloc[0]

            # Only keep the tuned version if it actually beat its own
            # untuned baseline -- tuning isn't guaranteed to win.
            if tuned_f1 > baseline_f1:
                tuned_name = f"{name} (Tuned)"
                fitted_models[tuned_name] = {"model": tuned_model, "train_time": 0.0}
                tuned_rows.append({
                    "Model": tuned_name, "Train Time (s)": 0.0,
                    "Validation Accuracy": round(tuned_acc, 4),
                    "Validation Macro F1": round(tuned_f1, 4),
                })
                logger.info(f"   ✅ {name}: tuned beat baseline ({tuned_f1:.4f} > {baseline_f1:.4f})")
            else:
                logger.info(f"   ⚠️ {name}: tuning did not beat baseline ({tuned_f1:.4f} <= {baseline_f1:.4f})")

        if tuned_rows:
            self.val_results = pd.concat(
                [self.val_results, pd.DataFrame(tuned_rows)], ignore_index=True
            ).sort_values(by="Validation Macro F1", ascending=False).reset_index(drop=True)

        best_row = self.val_results.iloc[0]
        self.best_model_name = best_row["Model"]
        self.best_model = fitted_models[self.best_model_name]["model"]

        logger.info(
            f"🏆 Best model on validation: {self.best_model_name} "
            f"(F1 macro={best_row['Validation Macro F1']:.4f})"
        )

        registered_model_name = config.registered_model_name(symbol, horizon,)

        with mlflow.start_run(run_name=f"{symbol}_{horizon}_{self.best_model_name}") as run:
            self.run_id = run.info.run_id
            mlflow.log_params({
                "model_name": self.best_model_name,
                "symbol": symbol,
                "horizon": horizon,
                "val_ratio": self.val_ratio,
                "test_ratio": self.test_ratio,
            })
            mlflow.log_metrics({
                "val_accuracy": best_row["Validation Accuracy"],
                "val_f1_macro": best_row["Validation Macro F1"],
                "train_time_s": best_row["Train Time (s)"],
            })

            # Bundle preprocessor + classifier into ONE deployable sklearn
            # Pipeline before logging -- a registered model version is one
            # opaque artifact, not a separately-tracked pair. This is what
            # Predictor.predict() in model_deployment.py calls directly.
            full_pipeline = Pipeline(steps=[
                ("preprocessor", self.preprocessor),
                ("classifier", self.best_model),
            ])
            mlflow.sklearn.log_model(
                full_pipeline,
                artifact_path="model",
                registered_model_name=registered_model_name,
                skops_trusted_types=["numpy.dtype"],
            )

        # Test set touched exactly once, after selection is final.
        self.test_metrics = evaluate_on_test(
            self.best_model, X_test_proc, y_test, run_id=self.run_id
        )

        return {
            "best_model_name": self.best_model_name,
            "best_model": self.best_model,
            "preprocessor": self.preprocessor,
            "val_results": self.val_results,
            "test_metrics": self.test_metrics,
            "run_id": self.run_id,
        }