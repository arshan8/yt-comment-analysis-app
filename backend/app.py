from flask import Flask, request, jsonify
from flask_cors import CORS
import mlflow
import numpy as np
import joblib
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# === Preprocessing function ===
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        comment = comment.lower().strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment

# === Load MLflow model + TF-IDF vectorizer ===
def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    mlflow.set_tracking_uri("http://13.60.38.114:5000")
    print(f"Loading model '{model_name}' version '{model_version}' from MLflow...")
    
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer("yt_chrome_plugin_model", "1", "./tfidf_vectorizer.pkl")

# === Predict route ===
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments')
    
    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        preprocessed_comments = [preprocess_comment(c) for c in comments]
        transformed_comments = vectorizer.transform(preprocessed_comments)

        feature_names = vectorizer.get_feature_names_out()
        X_df = pd.DataFrame(transformed_comments.toarray(), columns=feature_names)

        predictions = model.predict(X_df).tolist()

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, predictions)]
    return jsonify(response)

# === Run Flask ===
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
