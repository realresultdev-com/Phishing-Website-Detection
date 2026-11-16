"""
tests/test_reasons.py
---------------------
Tests for the plain-English explanation reason generator.
"""

import pytest
from src.explain.reasons import generate_reasons, format_risk_factor, FEATURE_EXPLANATIONS


def test_format_risk_factor():
    """Verify feature attribution formatting."""
    desc = format_risk_factor("is_ip_address", 1, 0.31, is_risk=True)
    assert "IP address" in desc
    assert "+0.31" in desc


def test_generate_reasons_phishing():
    """Verify plain-English explanation structure for phishing URLs."""
    shap_mock = {
        "shap_values": [0.35, 0.20, -0.05],
    }
    feature_names = ["is_ip_address", "sensitive_word_count", "uses_https"]
    raw_feats = {"is_ip_address": 1, "sensitive_word_count": 3, "uses_https": 1}

    res = generate_reasons(
        shap_local=shap_mock,
        lime_result=None,
        url_features=raw_feats,
        prediction_label="phishing",
        probability=0.92,
        feature_names=feature_names,
        top_k=3
    )

    assert res["verdict"] == "Phishing"
    assert "92.0%" in res["confidence"]
    assert len(res["risk_factors"]) >= 2
    assert len(res["protective_factors"]) >= 1
    assert "Flagged as phishing" in res["plain_english"]


def test_generate_reasons_legitimate():
    """Verify plain-English explanation structure for legitimate URLs."""
    shap_mock = {
        "shap_values": [-0.30, -0.15, 0.02],
    }
    feature_names = ["uses_https", "vowel_ratio", "count_hyphens"]
    raw_feats = {"uses_https": 1, "vowel_ratio": 0.45, "count_hyphens": 1}

    res = generate_reasons(
        shap_local=shap_mock,
        lime_result=None,
        url_features=raw_feats,
        prediction_label="legitimate",
        probability=0.08,
        feature_names=feature_names,
        top_k=3
    )

    assert res["verdict"] == "Legitimate"
    assert "Classified as legitimate" in res["plain_english"]
    assert len(res["protective_factors"]) >= 1


def test_empty_shap_fallback():
    """Handles empty or missing SHAP values safely without crash."""
    res = generate_reasons(
        shap_local=None,
        lime_result=None,
        url_features={"uses_https": 1, "is_ip_address": 0},
        prediction_label="legitimate",
        probability=0.10,
        feature_names=["uses_https", "is_ip_address"]
    )
    assert "verdict" in res
    assert "plain_english" in res
