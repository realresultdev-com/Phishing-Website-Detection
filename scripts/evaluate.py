"""
scripts/evaluate.py
-------------------
Stand-alone evaluation script.
Reloads the saved champion model and processed test split,
computes full metrics, regenerates evaluation curves, and prints the summary.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib

from src.data.load import load_config
from src.models.evaluate import evaluate_model, threshold_analysis
from src.models.predict import load_model, load_feature_schema


def main():
    print("====================================================================")
    print("  EXPLAINABLE AI PHISHING URL DETECTION — MODEL EVALUATION")
    print("====================================================================")

    config = load_config()
    artifacts_dir = PROJECT_ROOT / config.get("paths", {}).get("artifacts", "artifacts")
    data_proc_dir = PROJECT_ROOT / config.get("paths", {}).get("data_processed", "data/processed")

    # Load test split
    X_test_path = data_proc_dir / "X_test.npy"
    y_test_path = data_proc_dir / "y_test.npy"

    if not X_test_path.exists() or not y_test_path.exists():
        print(f"[ERROR] Processed test data not found in {data_proc_dir}.")
        print("Please run training first: python scripts/train.py")
        sys.exit(1)

    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)
    feature_names = load_feature_schema(artifacts_dir)

    print(f"Loaded test set: {len(X_test)} samples with {len(feature_names)} features.")

    # Load champion model
    try:
        pipeline = load_model(artifacts_dir=artifacts_dir)
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        sys.exit(1)

    model_name = getattr(pipeline.named_steps.get("clf", pipeline), "__class__", {}).__name__

    # Evaluate
    th = config.get("modeling", {}).get("threshold", 0.5)
    print(f"Evaluating {model_name} with decision threshold = {th:.2f}...")
    metrics = evaluate_model(
        pipeline=pipeline,
        X_test=X_test,
        y_test=y_test,
        model_name="champion_model",
        threshold=th,
        artifacts_dir=artifacts_dir
    )

    print("\n----------------- TEST SET EVALUATION METRICS -----------------")
    print(f"Accuracy               : {metrics['accuracy']:.4f}")
    print(f"Precision              : {metrics['precision']:.4f}")
    print(f"Recall (Sensitivity)   : {metrics['recall']:.4f}")
    print(f"F1-Score               : {metrics['f1']:.4f}")
    print(f"ROC-AUC Score          : {metrics['roc_auc']:.4f}")
    print(f"PR-AUC Score           : {metrics['pr_auc']:.4f}")
    print(f"Matthews Corr Coeff    : {metrics['mcc']:.4f}")
    print(f"False Positive Rate    : {metrics['fpr']:.4f}")
    print(f"False Negative Rate    : {metrics['fnr']:.4f} (Costly Security Hazard)")
    print(f"Median Inference Time  : {metrics['inference_latency_ms']:.2f} ms")
    print(f"Model Artifact Size    : {metrics['model_size_kb']:.1f} KB")
    print("---------------------------------------------------------------")

    # Threshold analysis
    print("\nRunning threshold sensitivity optimization...")
    th_res = threshold_analysis(pipeline, X_test, y_test, "champion", artifacts_dir)
    print(f"Optimal Threshold: {th_res['optimal_threshold']:.2f} (F1 = {th_res['optimal_f1']:.4f})")
    print("Evaluation complete. Plots updated in artifacts/plots/.")


if __name__ == "__main__":
    main()
