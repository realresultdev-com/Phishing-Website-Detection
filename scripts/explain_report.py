"""
scripts/explain_report.py
-------------------------
Comprehensive Explainable AI (XAI) report generation.
Produces:
1. Global SHAP analysis (beeswarm, bar summary, dependence plots for top-6 features)
2. Permutation feature importance and comparison against SHAP and model-native importance
3. Local explanations for demonstration URLs (SHAP waterfall, LIME explanation, Plain-English reasons)
4. Explanation quality evaluation:
   - SHAP-LIME agreement (top-k overlap, Spearman rank correlation)
   - Explanation stability under benign URL perturbations
   - Explanation fidelity via deletion/ablation testing
5. False Positive and False Negative error analysis with SHAP breakdown
"""

from __future__ import annotations

import sys
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib

from src.data.load import load_config
from src.features.url_features import extract_features, get_feature_names
from src.models.predict import load_model, load_feature_schema
from src.explain.shap_utils import (
    get_shap_explainer, compute_shap_values, plot_shap_summary,
    plot_shap_dependence, compute_permutation_importance,
    plot_permutation_importance, compute_shap_local, plot_shap_waterfall,
    compare_importances
)
from src.explain.lime_utils import (
    get_lime_explainer, explain_url_lime, plot_lime_explanation
)
from src.explain.reasons import generate_reasons
from src.explain.evaluate_explanations import (
    measure_shap_lime_agreement, measure_stability, measure_fidelity, analyze_errors
)


def main():
    print("====================================================================")
    print("  EXPLAINABLE AI (XAI) COMPREHENSIVE REPORT GENERATION")
    print("====================================================================")

    config = load_config()
    artifacts_dir = PROJECT_ROOT / config.get("paths", {}).get("artifacts", "artifacts")
    data_proc_dir = PROJECT_ROOT / config.get("paths", {}).get("data_processed", "data/processed")
    plots_dir = artifacts_dir / "plots"
    metrics_dir = artifacts_dir / "metrics"
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data and Model
    print("[XAI 1/6] Loading champion model and processed test dataset...")
    pipeline = load_model(artifacts_dir=artifacts_dir)
    feature_names = load_feature_schema(artifacts_dir=artifacts_dir)

    X_train = np.load(data_proc_dir / "X_train.npy")
    X_test = np.load(data_proc_dir / "X_test.npy")
    y_test = np.load(data_proc_dir / "y_test.npy")

    test_urls_path = data_proc_dir / "test_urls.csv"
    if test_urls_path.exists():
        urls_test = pd.read_csv(test_urls_path)["url"].tolist()
    else:
        urls_test = [f"url_sample_{i}" for i in range(len(X_test))]

    model_name = getattr(pipeline.named_steps.get("clf", pipeline), "__class__", {}).__name__
    print(f"Model: {model_name} | Test samples: {len(X_test)} | Features: {len(feature_names)}")

    # 2. Build Explainers
    print("\n[XAI 2/6] Initializing SHAP and LIME explainers...")
    bg_size = min(300, len(X_train))
    rng = np.random.RandomState(42)
    bg_idx = rng.choice(len(X_train), bg_size, replace=False)
    X_bg = X_train[bg_idx]

    shap_explainer = get_shap_explainer(pipeline, X_bg, model_name)
    lime_explainer = get_lime_explainer(X_train[:1000], feature_names)

    # 3. Global SHAP Analysis
    print("\n[XAI 3/6] Computing global SHAP explanations...")
    n_shap_global = min(config.get("explainability", {}).get("shap_sample_size", 400), len(X_test))
    eval_idx = rng.choice(len(X_test), n_shap_global, replace=False)
    X_eval = X_test[eval_idx]

    shap_cache_path = artifacts_dir / "shap_matrix_eval.npy"
    if shap_cache_path.exists():
        print("  Loading cached SHAP matrix...")
        shap_matrix = np.load(shap_cache_path)
    else:
        shap_matrix = compute_shap_values(shap_explainer, X_eval, model_name)
        np.save(shap_cache_path, shap_matrix)

    # Summary beeswarm and bar
    print("  Generating SHAP beeswarm plot...")
    plot_shap_summary(shap_matrix, X_eval, feature_names, str(plots_dir / "shap_summary_beeswarm.png"), plot_type="beeswarm")
    print("  Generating SHAP bar plot...")
    plot_shap_summary(shap_matrix, X_eval, feature_names, str(plots_dir / "shap_summary_bar.png"), plot_type="bar")

    # Dependence plots for top 6 features
    print("  Generating SHAP dependence plots for top 6 features...")
    plot_shap_dependence(shap_matrix, X_eval, feature_names, top_n=6, save_dir=str(plots_dir))

    # Permutation importance
    print("  Computing permutation importance...")
    perm_df = compute_permutation_importance(pipeline, X_eval, y_test[eval_idx], feature_names)
    plot_permutation_importance(perm_df, str(plots_dir / "permutation_importance.png"))

    # Importance comparison table
    shap_mean_abs = np.abs(shap_matrix).mean(axis=0)
    shap_df = pd.DataFrame({"feature": feature_names, "importance": shap_mean_abs})

    # Native importance if tree-based
    clf = pipeline.named_steps.get("clf", pipeline)
    if hasattr(clf, "feature_importances_"):
        native_df = pd.DataFrame({"feature": feature_names, "importance": clf.feature_importances_})
    elif hasattr(clf, "coef_"):
        native_df = pd.DataFrame({"feature": feature_names, "importance": np.abs(clf.coef_[0])})
    else:
        native_df = pd.DataFrame({"feature": feature_names, "importance": shap_mean_abs})

    comp_importances_df = compare_importances(shap_df, perm_df, native_df, feature_names)
    comp_importances_df.to_csv(metrics_dir / "feature_importance_comparison.csv", index=False)
    print("  Saved feature_importance_comparison.csv")

    # 4. Local Explanations Demonstration (5 representative URLs)
    print("\n[XAI 4/6] Generating local explanations for representative URLs...")
    demo_urls = [
        "https://www.paypal.com/signin",
        "http://paypal-account-verify.xyz/login.php",
        "https://www.google.com/search?q=cybersecurity",
        "http://192.168.1.50/admin/secure-banking-update.html",
        "https://github.com/security/vulnerability-alerts"
    ]

    demo_results = []
    for demo_url in demo_urls:
        raw_feats = extract_features(demo_url)
        vec = np.array([raw_feats.get(f, 0.0) for f in feature_names]).reshape(1, -1)
        prob = float(pipeline.predict_proba(vec)[0, 1])
        lbl = "phishing" if prob >= 0.5 else "legitimate"

        s_loc = compute_shap_local(shap_explainer, vec, feature_names, model_name)
        l_res = explain_url_lime(lime_explainer, vec, pipeline.predict_proba, num_features=10)
        reasons_dict = generate_reasons(
            shap_local=s_loc,
            lime_result=l_res,
            url_features=raw_feats,
            prediction_label=lbl,
            probability=prob,
            feature_names=feature_names,
            top_k=4
        )
        demo_results.append({
            "url": demo_url,
            "prediction": lbl,
            "probability": prob,
            "plain_english": reasons_dict["plain_english"]
        })
        print(f"  URL: {demo_url}")
        print(f"  -> {reasons_dict['plain_english']}\n")

    # 5. Explanation Quality Evaluation (Sample of 200 URLs)
    print("\n[XAI 5/6] Measuring explanation quality metrics (agreement, stability, fidelity)...")
    eval_n = min(config.get("explainability", {}).get("explanation_eval_sample", 200), len(X_eval))

    shap_ranks = []
    lime_ranks = []
    for i in range(eval_n):
        s_vals = shap_matrix[i]
        s_sorted = [feature_names[j] for j in np.argsort(np.abs(s_vals))[::-1]]
        shap_ranks.append(s_sorted)

        inst_vec = X_eval[i:i+1]
        l_res = explain_url_lime(lime_explainer, inst_vec, pipeline.predict_proba, num_features=15, num_samples=250)
        l_sorted = [f[0].split()[0] for f in l_res["feature_weights"]]
        lime_ranks.append(l_sorted)

    agreement_res = measure_shap_lime_agreement(shap_ranks, lime_ranks, top_k=5)
    print(f"  SHAP-LIME Top-5 Overlap    : {agreement_res['mean_top_k_overlap']:.3f}")
    print(f"  SHAP-LIME Rank Correlation : {agreement_res['mean_rank_correlation']:.3f}")

    # Stability test on a sample URL
    stability_res = measure_stability(
        model_pipeline=pipeline,
        url="http://secure-login-update.xyz/verify",
        feature_extractor_fn=extract_features,
        shap_local_fn=lambda v: compute_shap_local(shap_explainer, v, feature_names, model_name),
        feature_names=feature_names,
        n_perturbations=4
    )
    print(f"  Attribution Stability      : {stability_res['mean_attribution_stability']:.3f}")
    print(f"  Top-1 Feature Consistency  : {stability_res['top1_feature_consistency']:.3f}")

    # Fidelity test (feature ablation drop)
    fidelity_res = measure_fidelity(pipeline, X_eval, shap_matrix, k_values=[3, 5, 10])
    print(f"  Fidelity Deletion Drop (k=3): {fidelity_res['k3_mean_prob_drop']:.3f}")
    print(f"  Fidelity Deletion Drop (k=5): {fidelity_res['k5_mean_prob_drop']:.3f}")

    # 6. Error Analysis: 10 False Positives & 10 False Negatives
    print("\n[XAI 6/6] Analyzing False Positives and False Negatives...")
    error_analysis_res = analyze_errors(
        model_pipeline=pipeline,
        X_test=X_eval,
        y_test=y_test[eval_idx],
        urls=[urls_test[j] for j in eval_idx],
        feature_names=feature_names,
        shap_local_fn=lambda v: compute_shap_local(shap_explainer, v, feature_names, model_name),
        n_samples=10
    )
    print(f"  Identified {len(error_analysis_res['false_positives'])} False Positives and {len(error_analysis_res['false_negatives'])} False Negatives.")

    # Save complete XAI evaluation metrics
    xai_quality_report = {
        "shap_lime_agreement": agreement_res,
        "stability": stability_res,
        "fidelity": fidelity_res,
        "error_analysis_summary": {
            "fp_count": error_analysis_res["fp_count"],
            "fn_count": error_analysis_res["fn_count"],
            "sample_false_positives": error_analysis_res["false_positives"][:3],
            "sample_false_negatives": error_analysis_res["false_negatives"][:3]
        },
        "demo_explanations": demo_results
    }

    with open(metrics_dir / "explanation_quality.json", "w", encoding="utf-8") as f:
        json.dump(xai_quality_report, f, indent=2)
    print(f"\n✅ Explanation quality report successfully written to {metrics_dir / 'explanation_quality.json'}")


if __name__ == "__main__":
    main()
