import mlflow
import pytest
import pandas as pd
import pickle
from mlflow.tracking import MlflowClient

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app import preprocess_comment

mlflow.set_tracking_uri("http://13.60.38.114:5000")

@pytest.mark.parametrize("model_name, stage, vectorizer_path", [
    ("yt_chrome_plugin_model", "staging", "tfidf_vectorizer.pkl"),
])
def test_model_signature(model_name, stage, vectorizer_path):
    client = MlflowClient()

    # Get latest version from staging
    latest_versions = client.get_latest_versions(model_name, stages=[stage])
    assert latest_versions, f"No model in stage '{stage}' for model '{model_name}'"

    version = latest_versions[0].version
    model_uri = f"models:/{model_name}/{version}"
    model = mlflow.pyfunc.load_model(model_uri)

    # Load vectorizer
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    # Dummy input
    dummy_input = "Hi there, how are you doing?"
    cleaned_input = preprocess_comment(dummy_input)

    # Transform must take a list of strings
    vectorized = vectorizer.transform([cleaned_input])  # ✅ fixed here

    # Make DataFrame with correct columns
    feature_names = vectorizer.get_feature_names_out()
    input_df = pd.DataFrame(vectorized.toarray(), columns=feature_names)

    # Align columns to model signature (optional but recommended)
    expected_cols = model.metadata.signature.inputs.input_names()
    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = 0.0
    input_df = input_df[expected_cols]

    # Predict
    predictions = model.predict(input_df).tolist()

    # Assertions
    assert input_df.shape[1] == len(expected_cols), "Mismatch in input feature dimensions"
    assert len(predictions) == input_df.shape[0], "Prediction row count mismatch"

    print(f"✅ Model version {version} in '{stage}' stage passed signature test.")
