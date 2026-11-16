"""
src/models/predict.py
---------------------
Prediction inference interface for URL phishing classification.
Integrates feature extraction and trained model pipelines for single and batch predictions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Union

import joblib
import numpy as np
import pandas as pd

from src.features.url_features import extract_features, batch_extract_features, get_feature_names


def load_model(model_path: Union[str, Path, None] = None, artifacts_dir: Union[str, Path, None] = None) -> Any:
    """
    Load the trained model pipeline from artifacts/model.joblib.
    """
    if model_path is not None:
        p = Path(model_path)
    elif artifacts_dir is not None:
        p = Path(artifacts_dir) / "model.joblib"
    else:
        root = Path(__file__).resolve().parent.parent.parent
        p = root / "artifacts" / "model.joblib"

    if not p.exists():
        raise FileNotFoundError(f"Model file not found at {p}. Please train first using: python scripts/train.py")

    return joblib.load(p)


def load_feature_schema(artifacts_dir: Union[str, Path, None] = None) -> List[str]:
    """Load the ordered feature name list used during model training."""
    if artifacts_dir is not None:
        schema_path = Path(artifacts_dir) / "feature_schema.json"
    else:
        root = Path(__file__).resolve().parent.parent.parent
        schema_path = root / "artifacts" / "feature_schema.json"

    if schema_path.exists():
        with open(schema_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "feature_names" in data:
                return data["feature_names"]

    return get_feature_names()


def predict_url(
    url: str,
    pipeline: Any,
    feature_names: Optional[List[str]] = None,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Predict whether a single URL is phishing or legitimate.
    Does NOT visit or download the URL. Only extracts lexical/structural features.

    Returns
    -------
    dict with:
      - url: input url string
      - label: 'phishing' or 'legitimate'
      - is_phishing: boolean
      - probability: float (0.0 to 1.0)
      - threshold: decision threshold applied
      - features: dict of all extracted numerical features
      - feature_vector: list of float values matching model schema
    """
    if feature_names is None:
        feature_names = get_feature_names()

    # 1. Extract URL features
    raw_features = extract_features(url)

    # 2. Vectorize in exact model schema order
    feat_vector = [float(raw_features.get(f, 0.0)) for f in feature_names]
    X_vec = np.array(feat_vector).reshape(1, -1)

    # 3. Model inference
    if hasattr(pipeline, "predict_proba"):
        prob = float(pipeline.predict_proba(X_vec)[0, 1])
    else:
        pred_raw = pipeline.predict(X_vec)[0]
        prob = float(pred_raw)

    is_phish = bool(prob >= threshold)
    label = "phishing" if is_phish else "legitimate"

    return {
        "url": url,
        "label": label,
        "is_phishing": is_phish,
        "probability": prob,
        "threshold": threshold,
        "features": raw_features,
        "feature_vector": feat_vector
    }


def predict_batch(
    urls: List[str],
    pipeline: Any,
    feature_names: Optional[List[str]] = None,
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Predict a batch of URLs and return a results DataFrame.
    """
    if feature_names is None:
        feature_names = get_feature_names()

    # 1. Batch extract features
    feat_df = batch_extract_features(urls)

    # 2. Ensure all schema features exist
    for f in feature_names:
        if f not in feat_df.columns:
            feat_df[f] = 0.0

    X_mat = feat_df[feature_names].values

    # 3. Predict probabilities
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(X_mat)[:, 1]
    else:
        probs = pipeline.predict(X_mat)

    labels = ["phishing" if p >= threshold else "legitimate" for p in probs]

    results_df = pd.DataFrame({
        "url": urls,
        "verdict": labels,
        "phishing_probability": [round(float(p), 4) for p in probs],
        "risk_level": [
            "High" if p >= 0.75 else ("Medium" if p >= 0.5 else "Low")
            for p in probs
        ]
    })

    return results_df
