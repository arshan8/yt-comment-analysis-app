import pytest
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
from mlflow.tracking import MlflowClient

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app import preprocess_comment

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://13.60.38.114:5000")

@pytest.mark.parametrize("model_name, stage, holdout_data_path, vectorizer_path", [
    ("yt_chrome_plugin_model", "staging", "data/interim/test_processed.csv", "tfidf_vectorizer.pkl"),
])
def test_model_performance(model_name, stage, holdout_data_path, vectorizer_path):
    try:
        client = MlflowClient()
        latest_versions = client.get_latest_versions(model_name, stages=[stage])
        assert latest_versions, f"No model in stage '{stage}' for model '{model_name}'"

        version = latest_versions[0].version
        model_uri = f"models:/{model_name}/{version}"
        model = mlflow.pyfunc.load_model(model_uri)

        # Load vectorizer
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)

        # Load holdout test data
        holdout_data = pd.read_csv(holdout_data_path)
        X_holdout_raw = holdout_data.iloc[:, 0].fillna("")  # assuming text is in first column
        y_holdout = holdout_data.iloc[:, -1]                # assuming label is last column

        # Preprocess all text inputs
        X_holdout_cleaned = X_holdout_raw.apply(preprocess_comment)

        # Vectorize
        X_holdout_vectorized = vectorizer.transform(X_holdout_cleaned)
        feature_names = vectorizer.get_feature_names_out()
        X_holdout_df = pd.DataFrame(X_holdout_vectorized.toarray(), columns=feature_names)

        # Align with model signature
        expected_cols = model.metadata.signature.inputs.input_names()
        for col in expected_cols:
            if col not in X_holdout_df.columns:
                X_holdout_df[col] = 0.0
        X_holdout_df = X_holdout_df[expected_cols]

        # Predict
        y_pred = model.predict(X_holdout_df).tolist()

        # Metrics
        acc = accuracy_score(y_holdout, y_pred)
        prec = precision_score(y_holdout, y_pred, average='weighted', zero_division=1)
        rec = recall_score(y_holdout, y_pred, average='weighted', zero_division=1)
        f1 = f1_score(y_holdout, y_pred, average='weighted', zero_division=1)

        # Thresholds
        assert acc >= 0.40, f"Accuracy too low: {acc}"
        assert prec >= 0.40, f"Precision too low: {prec}"
        assert rec >= 0.40, f"Recall too low: {rec}"
        assert f1 >= 0.40, f"F1 Score too low: {f1}"

        print(f"✅ Model version {version} passed performance test - acc: {acc:.2f}, prec: {prec:.2f}, rec: {rec:.2f}, f1: {f1:.2f}")

    except Exception as e:
        pytest.fail(f"❌ Model performance test failed: {e}")
