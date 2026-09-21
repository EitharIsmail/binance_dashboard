import mlflow
from mlflow import MlflowClient

# --------------------------------------------------
# Source: trained model artifact
# --------------------------------------------------
SOURCE_MODEL_PATH = (
    "3_Pipeline/mlartifacts/1/models/"
    "m-0e9ae841bf80496a81b5ce8594e11fe7/artifacts"
)

# --------------------------------------------------
# Destination: Docker MLflow
# --------------------------------------------------
DEPLOYMENT_MLFLOW_URI = "http://localhost:5000"

REGISTERED_MODEL_NAME = "BTCUSDT_30m_classifier"

# Original training metrics
VAL_F1 = 0.4033
TEST_F1 = 0.3627771894616057


# --------------------------------------------------
# 1. Load the trained pipeline
# --------------------------------------------------
print("Loading trained model...")

model = mlflow.sklearn.load_model(SOURCE_MODEL_PATH)

print("Model loaded:")
print(model)


# --------------------------------------------------
# 2. Connect to deployment MLflow
# --------------------------------------------------
print("\nConnecting to deployment MLflow...")

mlflow.set_tracking_uri(DEPLOYMENT_MLFLOW_URI)

print("Tracking URI:", mlflow.get_tracking_uri())


# --------------------------------------------------
# 3. Create an experiment for the deployment
# --------------------------------------------------
experiment_name = "binance-deployment"

experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
else:
    experiment_id = experiment.experiment_id

print("Experiment ID:", experiment_id)


# --------------------------------------------------
# 4. Start a deployment run
# --------------------------------------------------
with mlflow.start_run(
    experiment_id=experiment_id,
    run_name="BTCUSDT_30m_AdaBoost_deployment",
):

    mlflow.log_param("model_name", "AdaBoost")
    mlflow.log_param("symbol", "BTCUSDT")
    mlflow.log_param("horizon", "30m")
    mlflow.log_param("source_run_id", "70bad8e64ddc445f8262838129e6b182")

    mlflow.log_metric("val_f1_macro", VAL_F1)
    mlflow.log_metric("test_f1_macro", TEST_F1)

    mlflow.set_tag("deployment_ready", "true")
    mlflow.set_tag("source_run_id", "70bad8e64ddc445f8262838129e6b182")
    mlflow.set_tag("model_name", "AdaBoost")
    mlflow.set_tag("symbol", "BTCUSDT")
    mlflow.set_tag("horizon", "30m")

    # --------------------------------------------------
    # 5. Register the model
    # --------------------------------------------------
    print("\nRegistering model...")

    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        registered_model_name=REGISTERED_MODEL_NAME,
        skops_trusted_types=["numpy.dtype"],
    )

print("\nModel registered successfully.")