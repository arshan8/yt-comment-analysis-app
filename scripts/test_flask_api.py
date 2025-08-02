import pytest
import requests
import json

BASE_URL = "http://localhost:5000"  # Replace with your deployed URL if needed

def test_predict_endpoint():
    data = {
        "comments": ["This is a great product!", "Not worth the money.", "It's okay."]
    }
    response = requests.post(f"{BASE_URL}/predict", json=data)
    assert response.status_code == 200
    assert isinstance(response.json(), list)

