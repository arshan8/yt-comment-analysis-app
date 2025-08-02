
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.pyfunc
import pytest

mlflow.set_tracking_uri("http://13.60.38.114:5000")
client = MlflowClient()
model_name = "yt_chrome_plugin_model"

# ✅ Step 1: Transition latest version to Staging
latest_version = 4  # You can set this manually or fetch dynamically
client.transition_model_version_stage(
    name=model_name,
    version=latest_version,
    stage="Staging",
    archive_existing_versions=False
)
print(f"✅ Version {latest_version} transitioned to 'Staging'")

# ✅ Step 2: Test loading the model from Staging
@pytest.mark.parametrize("model_name, stage", [(model_name, "Staging")])
def test_load_latest_staging_model(model_name, stage):
    latest_versions = client.get_latest_versions(model_name, stages=[stage])
    version = latest_versions[0].version if latest_versions else None

    assert version is not None, f"No model in stage '{stage}' for model '{model_name}'"

    try:
        model_uri = f"models:/{model_name}/{stage}"
        model = mlflow.pyfunc.load_model(model_uri)
        assert model is not None, "Model failed to load"
        print(f"✅ Model '{model_name}' version {version} loaded successfully from stage '{stage}'")
    except Exception as e:
        pytest.fail(f"Model loading failed with error: {e}")

# Run the test manually
test_load_latest_staging_model(model_name, "Staging")
