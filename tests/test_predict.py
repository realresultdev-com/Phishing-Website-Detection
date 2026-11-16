"""
tests/test_predict.py
---------------------
Tests for URL prediction and batch inference logic.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from src.features.url_features import get_feature_names
from src.models.predict import predict_url, predict_batch


class MockPipeline:
    def __init__(self, prob=0.85):
        self.prob = prob

    def predict_proba(self, X):
        probs = np.zeros((len(X), 2))
        probs[:, 0] = 1.0 - self.prob
        probs[:, 1] = self.prob
        return probs

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def test_predict_url_structure():
    """Verify single URL prediction dictionary schema."""
    pipe = MockPipeline(prob=0.91)
    res = predict_url("http://phish-login.tk/verify", pipe, get_feature_names(), threshold=0.5)

    assert res["url"] == "http://phish-login.tk/verify"
    assert res["label"] == "phishing"
    assert res["is_phishing"] is True
    assert 0.0 <= res["probability"] <= 1.0
    assert res["probability"] == 0.91
    assert "features" in res
    assert "feature_vector" in res
    assert len(res["feature_vector"]) == len(get_feature_names())


def test_predict_url_threshold_rule():
    """Verify threshold classification edge conditions."""
    pipe = MockPipeline(prob=0.45)
    # With threshold 0.5 -> legitimate
    res1 = predict_url("https://example.com", pipe, get_feature_names(), threshold=0.5)
    assert res1["label"] == "legitimate"
    assert res1["is_phishing"] is False

    # With threshold 0.4 -> phishing
    res2 = predict_url("https://example.com", pipe, get_feature_names(), threshold=0.4)
    assert res2["label"] == "phishing"
    assert res2["is_phishing"] is True


def test_predict_batch():
    """Verify batch prediction DataFrame output."""
    pipe = MockPipeline(prob=0.78)
    urls = [
        "https://google.com",
        "http://suspicious-bank-login.xyz/auth",
        "https://github.com/torvalds/linux"
    ]
    df = predict_batch(urls, pipe, get_feature_names(), threshold=0.5)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert set(df.columns) == {"url", "verdict", "phishing_probability", "risk_level"}
    assert all(df["verdict"] == "phishing")
    assert all(df["risk_level"] == "High")


def test_predict_no_crash_on_malformed_url():
    """Verify robust inference on unusual/malformed URLs."""
    pipe = MockPipeline(prob=0.20)
    res = predict_url(":::not a valid url;;;", pipe, get_feature_names(), threshold=0.5)
    assert res["label"] == "legitimate"
    assert isinstance(res["probability"], float)
