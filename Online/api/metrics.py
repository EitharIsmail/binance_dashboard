from prometheus_client import Counter, Gauge, Histogram, Info


# ============================================================
# HTTP REQUEST METRICS
# ============================================================

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=[
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1,
        2.5,
        5,
        10,
    ],
)

ACTIVE_REQUESTS = Gauge(
    "active_requests",
    "Number of HTTP requests currently being processed",
)


# ============================================================
# PREDICTION METRICS
# ============================================================

PREDICTION_COUNT = Counter(
    "predictions_total",
    "Total number of predictions made",
)

PREDICTION_LATENCY = Histogram(
    "prediction_duration_seconds",
    "Time required to compute a prediction",
    buckets=[
        0.001,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1,
    ],
)

PREDICTION_RESULT_COUNT = Counter(
    "prediction_results_total",
    "Total number of predictions by direction",
    ["direction"],
)


# ============================================================
# MODEL INFORMATION
# ============================================================

MODEL_INFO = Info(
    "model",
    "Information about the currently loaded Binance model",
)

# ============================================================
# OFFLINE MODEL EVALUATION METRICS FROM MLFLOW
# ============================================================

MODEL_VALIDATION_ACCURACY = Gauge(
    "model_validation_accuracy",
    "Validation accuracy of the deployed model",
    ["symbol", "horizon", "version"],
)

MODEL_TEST_ACCURACY = Gauge(
    "model_test_accuracy",
    "Test accuracy of the deployed model",
    ["symbol", "horizon", "version"],
)

MODEL_VALIDATION_F1 = Gauge(
    "model_validation_f1_macro",
    "Validation macro F1 score of the deployed model",
    ["symbol", "horizon", "version"],
)

MODEL_TEST_F1 = Gauge(
    "model_test_f1_macro",
    "Test macro F1 score of the deployed model",
    ["symbol", "horizon", "version"],
)