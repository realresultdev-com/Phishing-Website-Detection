"""
src/models/evaluate.py
----------------------
Evaluation module for Phishing URL Detection models.
Calculates all project metrics:
- Accuracy, Precision, Recall, F1
- ROC-AUC, PR-AUC, MCC (Matthews Correlation Coefficient)
- Confusion matrix, False Positive Rate (FPR), False Negative Rate (FNR)
- Inference latency per URL (median over repeated single-sample passes)
- Model size in KB
- Generates publication-ready evaluation plots:
  * Overlaid ROC curves
  * Overlaid Precision-Recall curves
  * Model comparison bar charts
  * Calibration reliability curves
  * Decision threshold analysis curve
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, matthews_corrcoef,
    confusion_matrix, roc_curve
)
from sklearn.calibration import calibration_curve


def evaluate_model(
    pipeline: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    threshold: float = 0.5,
    artifacts_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive metrics for a given model pipeline on the test set.
    """
    # Prediction probabilities
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(X_test)[:, 1]
    else:
        probs = pipeline.predict(X_test)
        probs = (probs - probs.min()) / (probs.max() - probs.min() + 1e-9)

    preds = (probs >= threshold).astype(int)

    # Core classification metrics
    acc = float(accuracy_score(y_test, preds))
    prec = float(precision_score(y_test, preds, zero_division=0))
    rec = float(recall_score(y_test, preds, zero_division=0))
    f1 = float(f1_score(y_test, preds, zero_division=0))
    mcc = float(matthews_corrcoef(y_test, preds))

    try:
        roc_auc = float(roc_auc_score(y_test, probs))
    except Exception:
        roc_auc = 0.5

    try:
        p_vals, r_vals, _ = precision_recall_curve(y_test, probs)
        pr_auc = float(auc(r_vals, p_vals))
    except Exception:
        pr_auc = 0.5

    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    # Measure inference latency (single URL inference, repeated 100 times)
    sample_vec = X_test[0:1]
    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        pipeline.predict_proba(sample_vec)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    median_latency_ms = float(np.median(latencies))

    # Model size in KB
    model_size_kb = 0.0
    if artifacts_dir:
        model_file = artifacts_dir / f"model_{model_name}.joblib"
        if not model_file.exists():
            model_file = artifacts_dir / "model.joblib"
        if model_file.exists():
            model_size_kb = float(model_file.stat().st_size / 1024.0)

    # Save individual confusion matrix plot
    if artifacts_dir:
        plots_dir = artifacts_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
            xticklabels=["Legitimate", "Phishing"],
            yticklabels=["Legitimate", "Phishing"]
        )
        ax.set_title(f"Confusion Matrix — {model_name.upper()}\n(Threshold={threshold:.2f})")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(plots_dir / f"cm_{model_name}.png", dpi=200)
        plt.close(fig)

    return {
        "model_name": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "mcc": mcc,
        "fpr": fpr,
        "fnr": fnr,
        "confusion_matrix": cm.tolist(),
        "inference_latency_ms": median_latency_ms,
        "model_size_kb": model_size_kb,
        "threshold": threshold,
        "probs": probs.tolist()
    }


def plot_all_models_comparison(
    eval_results: Dict[str, Dict[str, Any]],
    y_test: np.ndarray,
    artifacts_dir: Path
) -> None:
    """Generate and save comparison plots across all trained models."""
    plots_dir = artifacts_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Overlaid ROC Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for m_name, res in eval_results.items():
        probs = np.array(res["probs"])
        fpr_curve, tpr_curve, _ = roc_curve(y_test, probs)
        ax.plot(fpr_curve, tpr_curve, label=f"{m_name} (AUC = {res['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (0.50)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plots_dir / "roc_curves.png", dpi=200)
    plt.close(fig)

    # 2. Overlaid Precision-Recall Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for m_name, res in eval_results.items():
        probs = np.array(res["probs"])
        p_curve, r_curve, _ = precision_recall_curve(y_test, probs)
        ax.plot(r_curve, p_curve, label=f"{m_name} (PR-AUC = {res['pr_auc']:.3f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves Comparison")
    ax.legend(loc="lower left")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plots_dir / "pr_curves.png", dpi=200)
    plt.close(fig)

    # 3. Model comparison bar chart (F1, ROC-AUC, Recall)
    models = list(eval_results.keys())
    f1_scores = [eval_results[m]["f1"] for m in models]
    roc_scores = [eval_results[m]["roc_auc"] for m in models]
    recall_scores = [eval_results[m]["recall"] for m in models]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, f1_scores, width, label="F1-Score", color="#1f77b4")
    ax.bar(x, roc_scores, width, label="ROC-AUC", color="#2ca02c")
    ax.bar(x + width, recall_scores, width, label="Recall", color="#ff7f0e")

    ax.set_ylabel("Score")
    ax.set_title("Model Performance Benchmarking Across Primary Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in models], rotation=15)
    ax.set_ylim([0.0, 1.1])
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plots_dir / "model_comparison.png", dpi=200)
    plt.close(fig)

    # 4. Calibration reliability curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for m_name, res in eval_results.items():
        probs = np.array(res["probs"])
        try:
            prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)
            ax.plot(prob_pred, prob_true, marker="o", label=m_name)
        except Exception:
            pass
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly Calibrated")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives (Observed)")
    ax.set_title("Calibration Reliability Curves")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plots_dir / "calibration.png", dpi=200)
    plt.close(fig)


def threshold_analysis(
    pipeline: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    artifacts_dir: Path
) -> Dict[str, Any]:
    """
    Analyze performance across classification thresholds from 0.05 to 0.95.
    Identifies the optimal threshold balancing high recall (low false negative rate)
    with precision.
    """
    probs = pipeline.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0.05, 0.95, 37)

    precisions = []
    recalls = []
    f1s = []
    fnrs = []
    fprs = []

    for th in thresholds:
        preds = (probs >= th).astype(int)
        p = precision_score(y_test, preds, zero_division=0)
        r = recall_score(y_test, preds, zero_division=0)
        f = f1_score(y_test, preds, zero_division=0)
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
        fnrs.append(fnr)
        fprs.append(fpr)

    # Best threshold by F1
    best_idx = int(np.argmax(f1s))
    best_th = float(thresholds[best_idx])

    plots_dir = artifacts_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, f1s, label="F1-Score", color="blue", lw=2)
    ax.plot(thresholds, precisions, label="Precision", color="green", linestyle="--")
    ax.plot(thresholds, recalls, label="Recall (1 - FNR)", color="orange", linestyle="--")
    ax.plot(thresholds, fnrs, label="False Negative Rate (Costly Error)", color="red", linestyle=":")
    ax.axvline(best_th, color="purple", linestyle="-.", label=f"Optimal F1 Threshold ({best_th:.2f})")

    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Metric Value")
    ax.set_title(f"Threshold Optimization Analysis ({model_name.upper()})")
    ax.legend(loc="center left")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plots_dir / "threshold_analysis.png", dpi=200)
    plt.close(fig)

    return {
        "optimal_threshold": best_th,
        "optimal_f1": float(f1s[best_idx]),
        "optimal_precision": float(precisions[best_idx]),
        "optimal_recall": float(recalls[best_idx]),
        "optimal_fnr": float(fnrs[best_idx])
    }


def save_comparison_table(
    eval_results: Dict[str, Dict[str, Any]],
    artifacts_dir: Path
) -> pd.DataFrame:
    """Save metrics comparison table as CSV and Markdown formats."""
    metrics_dir = artifacts_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for m_name, res in eval_results.items():
        rows.append({
            "Model": m_name.replace("_", " ").title(),
            "Accuracy": round(res["accuracy"], 4),
            "Precision": round(res["precision"], 4),
            "Recall": round(res["recall"], 4),
            "F1-Score": round(res["f1"], 4),
            "ROC-AUC": round(res["roc_auc"], 4),
            "PR-AUC": round(res["pr_auc"], 4),
            "MCC": round(res["mcc"], 4),
            "FPR": round(res["fpr"], 4),
            "FNR": round(res["fnr"], 4),
            "Latency (ms)": round(res["inference_latency_ms"], 2),
            "Size (KB)": round(res["model_size_kb"], 1)
        })

    df = pd.DataFrame(rows)
    df.sort_values(by="ROC-AUC", ascending=False, inplace=True)

    csv_path = metrics_dir / "model_comparison.csv"
    md_path = metrics_dir / "model_comparison.md"
    json_path = metrics_dir / "model_comparison.json"

    df.to_csv(csv_path, index=False)
    try:
        df.to_markdown(md_path, index=False)
    except Exception:
        # Fallback manual markdown table generator
        headers = list(df.columns)
        lines = ["| " + " | ".join(headers) + " |"]
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(val) for val in row.values) + " |")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    # Save clean JSON version (without raw probs)
    clean_dict = {}
    for k, v in eval_results.items():
        v_copy = dict(v)
        v_copy.pop("probs", None)
        clean_dict[k] = v_copy
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(clean_dict, f, indent=2)

    print(f"\n=================== MODEL COMPARISON ===================")
    print(df.to_string(index=False))
    print("========================================================\n")

    return df
