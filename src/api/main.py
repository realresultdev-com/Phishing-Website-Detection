"""
src/api/main.py
---------------
FastAPI REST API for Explainable AI-Based Phishing URL Detection.
Provides endpoints for:
- Health checking (/health)
- Model metadata and benchmark information (/model-info)
- Single URL prediction (/predict)
- Single URL prediction with SHAP/LIME plain-English explanation (/explain)
- Batch URL processing (/predict/batch)
"""

from __future__ import annotations

import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from src.features.url_features import extract_features, get_feature_names, batch_extract_features
from src.models.predict import load_model, load_feature_schema, predict_url
from src.explain.shap_utils import get_shap_explainer, compute_shap_local, plot_shap_waterfall
from src.explain.lime_utils import get_lime_explainer, explain_url_lime, plot_lime_explanation
from src.explain.reasons import generate_reasons


# ── Pydantic Request & Response Schemas ────────────────────────────────────────

class URLRequest(BaseModel):
    url: str = Field(..., description="Target URL string to inspect", max_length=2000)
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Decision threshold for phishing classification")

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("URL string cannot be empty or purely whitespace")
        return s


class BatchURLRequest(BaseModel):
    urls: List[str] = Field(..., max_length=1000, description="List of URL strings to analyze (max 1000)")
    threshold: float = Field(0.5, ge=0.0, le=1.0)

    @field_validator("urls")
    @classmethod
    def validate_urls(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("URL list cannot be empty")
        return [u.strip() for u in v if u.strip()]


class PredictionResponse(BaseModel):
    url: str
    label: str
    is_phishing: bool
    probability: float
    threshold: float
    inference_time_ms: float


class FeatureAttribution(BaseModel):
    feature: str
    value: Any
    shap_value: float
    direction: str
    explanation: str


class ExplanationResponse(BaseModel):
    url: str
    label: str
    is_phishing: bool
    probability: float
    threshold: float
    summary: str
    plain_english: str
    risk_factors: List[FeatureAttribution]
    protective_factors: List[FeatureAttribution]
    shap_waterfall_b64: Optional[str] = None
    lime_chart_b64: Optional[str] = None
    inference_time_ms: float


class BatchItemResult(BaseModel):
    url: str
    verdict: str
    phishing_probability: float
    risk_level: str


class BatchPredictionResponse(BaseModel):
    results: List[BatchItemResult]
    total: int
    phishing_count: int
    legitimate_count: int
    processing_time_ms: float


class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    feature_count: int
    default_threshold: float
    training_dataset: str
    metrics: Dict[str, Any]


# ── FastAPI Application Initialisation ─────────────────────────────────────────

app = FastAPI(
    title="Phishing URL Detection & Explainability API",
    description="Explainable AI framework predicting website phishing risk based exclusively on lexical and structural URL attributes (no network visits).",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory cache for models and explainers
_STATE: Dict[str, Any] = {
    "model": None,
    "feature_names": None,
    "shap_explainer": None,
    "lime_explainer": None,
    "metrics": {}
}


def _get_model_or_raise():
    """Retrieve or lazily initialize the trained model pipeline."""
    if _STATE["model"] is None:
        try:
            _STATE["model"] = load_model(artifacts_dir=PROJECT_ROOT / "artifacts")
            _STATE["feature_names"] = load_feature_schema(artifacts_dir=PROJECT_ROOT / "artifacts")

            # Load metrics summary if present
            metrics_json = PROJECT_ROOT / "artifacts" / "metrics" / "model_comparison.json"
            if metrics_json.exists():
                with open(metrics_json, "r", encoding="utf-8") as f:
                    _STATE["metrics"] = json.load(f)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP(503),
                detail=f"Model artifacts not available. Please run model training first: 'python scripts/train.py'. Error: {e}"
            )
    return _STATE["model"], _STATE["feature_names"]


def _get_explainers():
    """Lazily initialize SHAP and LIME explainers."""
    if _STATE["shap_explainer"] is None or _STATE["lime_explainer"] is None:
        model, feature_names = _get_model_or_raise()
        model_name = getattr(model.named_steps.get("clf", model), "__class__", {}).__name__

        # Load background data for explainers
        proc_dir = PROJECT_ROOT / "data" / "processed"
        X_tr_path = proc_dir / "X_train.npy"
        if X_tr_path.exists():
            X_train = np.load(X_tr_path)
            bg_data = X_train[:200]
        else:
            bg_data = np.zeros((50, len(feature_names)))

        _STATE["shap_explainer"] = get_shap_explainer(model, bg_data, model_name)
        _STATE["lime_explainer"] = get_lime_explainer(bg_data, feature_names)

    return _STATE["shap_explainer"], _STATE["lime_explainer"]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    """Health check endpoint checking service availability and model status."""
    has_model = (PROJECT_ROOT / "artifacts" / "model.joblib").exists()
    return {
        "status": "healthy",
        "service": "phishing-xai-api",
        "model_loaded": has_model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }


@app.get("/model-info", response_model=ModelInfoResponse, tags=["System"])
def model_info():
    """Retrieve metadata, feature count, and benchmark metrics for the active champion model."""
    model, feature_names = _get_model_or_raise()
    clf = model.named_steps.get("clf", model)
    model_name = getattr(clf, "__class__", {}).__name__

    return ModelInfoResponse(
        model_name=model_name,
        version="1.0.0",
        feature_count=len(feature_names),
        default_threshold=0.5,
        training_dataset="UCI PhiUSIIL Phishing URL Dataset (Grouped by Registered Domain)",
        metrics=_STATE.get("metrics", {})
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(request: URLRequest):
    """
    Classify a single URL as phishing or legitimate using lexical and structural features.
    Guaranteed safe: the target URL is never visited or queried over the network.
    """
    t0 = time.perf_counter()
    model, feature_names = _get_model_or_raise()

    res = predict_url(request.url, model, feature_names, threshold=request.threshold)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    return PredictionResponse(
        url=res["url"],
        label=res["label"],
        is_phishing=res["is_phishing"],
        probability=res["probability"],
        threshold=res["threshold"],
        inference_time_ms=round(latency_ms, 2)
    )


@app.post("/explain", response_model=ExplanationResponse, tags=["Explainability"])
def explain(request: URLRequest):
    """
    Perform local Explainable AI diagnosis for a URL.
    Returns:
    - Predicted risk label and probability
    - Plain-English bullet list of top risk-increasing and protective factors
    - SHAP waterfall attribution and LIME visualization charts (base64 PNG)
    """
    t0 = time.perf_counter()
    model, feature_names = _get_model_or_raise()
    model_name = getattr(model.named_steps.get("clf", model), "__class__", {}).__name__

    # 1. Feature extraction & prediction
    pred_res = predict_url(request.url, model, feature_names, threshold=request.threshold)
    raw_feats = pred_res["features"]
    vec = np.array(pred_res["feature_vector"]).reshape(1, -1)

    # 2. Compute local SHAP and LIME
    shap_b64 = None
    lime_b64 = None
    s_loc = None
    l_res = None

    try:
        shap_exp, lime_exp = _get_explainers()
        s_loc = compute_shap_local(shap_exp, vec, feature_names, model_name)
        l_res = explain_url_lime(lime_exp, vec, model.predict_proba, num_features=10)

        shap_b64 = plot_shap_waterfall(
            s_loc["shap_values"], s_loc["base_value"], feature_names,
            s_loc["feature_values"], pred_res["probability"]
        )
        lime_b64 = plot_lime_explanation(l_res)
    except Exception as e:
        print(f"[EXPLAIN ERROR] Explainer error: {e}")

    # 3. Generate plain-English reasons
    reasons = generate_reasons(
        shap_local=s_loc,
        lime_result=l_res,
        url_features=raw_feats,
        prediction_label=pred_res["label"],
        probability=pred_res["probability"],
        feature_names=feature_names,
        top_k=5
    )

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return ExplanationResponse(
        url=request.url,
        label=pred_res["label"],
        is_phishing=pred_res["is_phishing"],
        probability=pred_res["probability"],
        threshold=pred_res["threshold"],
        summary=reasons["summary"],
        plain_english=reasons["plain_english"],
        risk_factors=[
            FeatureAttribution(
                feature=rf["feature"],
                value=rf["value"],
                shap_value=rf["shap"],
                direction=rf["direction"],
                explanation=rf["explanation"]
            )
            for rf in reasons["risk_factors"]
        ],
        protective_factors=[
            FeatureAttribution(
                feature=pf["feature"],
                value=pf["value"],
                shap_value=pf["shap"],
                direction=pf["direction"],
                explanation=pf["explanation"]
            )
            for pf in reasons["protective_factors"]
        ],
        shap_waterfall_b64=shap_b64,
        lime_chart_b64=lime_b64,
        inference_time_ms=round(latency_ms, 2)
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch_endpoint(request: BatchURLRequest):
    """
    Process up to 1000 URLs in batch.
    Extracts features concurrently and computes predictions with high throughput.
    """
    t0 = time.perf_counter()
    model, feature_names = _get_model_or_raise()

    feat_df = batch_extract_features(request.urls)
    for f in feature_names:
        if f not in feat_df.columns:
            feat_df[f] = 0.0

    X_mat = feat_df[feature_names].values
    probs = model.predict_proba(X_mat)[:, 1]

    results = []
    phishing_cnt = 0
    legit_cnt = 0

    for url, prob in zip(request.urls, probs):
        is_phish = prob >= request.threshold
        verdict = "phishing" if is_phish else "legitimate"
        if is_phish:
            phishing_cnt += 1
        else:
            legit_cnt += 1

        risk_level = "High" if prob >= 0.75 else ("Medium" if prob >= 0.50 else "Low")
        results.append(BatchItemResult(
            url=url,
            verdict=verdict,
            phishing_probability=round(float(prob), 4),
            risk_level=risk_level
        ))

    proc_time_ms = (time.perf_counter() - t0) * 1000.0

    return BatchPredictionResponse(
        results=results,
        total=len(request.urls),
        phishing_count=phishing_cnt,
        legitimate_count=legit_cnt,
        processing_time_ms=round(proc_time_ms, 2)
    )
