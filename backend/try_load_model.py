import mlflow

# Set MLflow tracking URI (remote server)
mlflow.set_tracking_uri("http://13.60.38.114:5000")

# Load model from registry
def load_model_from_registry(model_name: str, model_version: str):
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    return model

# Test function
if __name__ == "__main__":
    model_name = "yt_chrome_plugin_model"   # change if needed
    model_version = "1"                     # change to your latest version

    try:
        model = load_model_from_registry(model_name, model_version)
        print(f"✅ Successfully loaded model: {model_name} (version {model_version})")
        # Optionally test a dummy input if you're sure of input schema:
        # sample = {"text": ["this video is amazing"]}
        # prediction = model.predict(sample)
        # print(prediction)
    except Exception as e:
        print("❌ Failed to load model from registry.")
        print(e)
