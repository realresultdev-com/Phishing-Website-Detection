"""
tests/test_api.py
-----------------
API unit and integration tests using FastAPI TestClient.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify /health responds with 200 and expected schema."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert "timestamp" in data


def test_docs_accessible():
    """Verify OpenAPI documentation is reachable."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_predict_empty_url():
    """Empty or whitespace URL must trigger 422 unprocessable entity."""
    response = client.post("/predict", json={"url": "   ", "threshold": 0.5})
    assert response.status_code == 422


def test_predict_url_length_limit():
    """URLs exceeding 2000 characters must trigger 422."""
    long_url = "https://example.com/" + "a" * 2005
    response = client.post("/predict", json={"url": long_url, "threshold": 0.5})
    assert response.status_code == 422


def test_predict_batch_empty_list():
    """Empty URL list in batch must trigger 422."""
    response = client.post("/predict/batch", json={"urls": [], "threshold": 0.5})
    assert response.status_code == 422


def test_predict_batch_too_many_urls():
    """Batch requests exceeding 1000 URLs must trigger 422."""
    urls = [f"https://example.com/{i}" for i in range(1005)]
    response = client.post("/predict/batch", json={"urls": urls, "threshold": 0.5})
    assert response.status_code == 422


def test_predict_and_explain_endpoints():
    """
    Test /predict and /explain endpoints if model artifacts exist.
    """
    response_health = client.get("/health").json()
    if not response_health["model_loaded"]:
        pytest.skip("Model not trained yet — skipping prediction integration test")

    # Single prediction
    res_pred = client.post("/predict", json={"url": "http://paypal-verification.xyz/login.php", "threshold": 0.5})
    assert res_pred.status_code == 200
    data_pred = res_pred.json()
    assert "label" in data_pred
    assert "probability" in data_pred
    assert 0.0 <= data_pred["probability"] <= 1.0
    assert "inference_time_ms" in data_pred

    # Batch prediction
    res_batch = client.post("/predict/batch", json={
        "urls": [
            "https://www.google.com",
            "http://192.168.1.1/admin/login.php",
            "https://github.com"
        ],
        "threshold": 0.5
    })
    assert res_batch.status_code == 200
    data_batch = res_batch.json()
    assert data_batch["total"] == 3
    assert len(data_batch["results"]) == 3
    assert data_batch["phishing_count"] + data_batch["legitimate_count"] == 3

    # Explanation
    res_exp = client.post("/explain", json={"url": "http://192.168.1.1/admin/login.php", "threshold": 0.5})
    assert res_exp.status_code == 200
    data_exp = res_exp.json()
    assert "plain_english" in data_exp
    assert "risk_factors" in data_exp
    assert "summary" in data_exp
