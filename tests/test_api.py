import pytest
from fastapi.testclient import TestClient
from app import app, load_model

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_predict_endpoint_no_model():
    # If we haven't loaded the model, it should return an error gracefully
    response = client.post("/predict", json={"text": "mera order kahan hai"}, headers={"X-API-Key": "intentify-secret-key"})
    assert response.status_code == 200
    data = response.json()
    assert "error" in data or "predicted_intent" in data

def test_predict_endpoint_auth_fail():
    response = client.post("/predict", json={"text": "mera order kahan hai"})
    assert response.status_code == 403
