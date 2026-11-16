"""
scripts/train.py
----------------
End-to-end training pipeline for Phishing URL Detection.
Orchestrates:
1. Data loading (with synthetic fallback if raw data is missing)
2. Feature extraction (URL lexical/structural features only)
3. Domain-grouped leak-free train/val/test splitting
4. Random split comparison to demonstrate data leakage impact
5. Training & tuning of 6 classifiers
6. Comprehensive evaluation and benchmarking
7. Saving champion model and feature schema
"""

from __future__ import annotations

import sys
import time
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score

from src.data.load import load_config, load_dataset
from src.features.url_features import batch_extract_features, get_feature_names
from src.data.split import group_aware_split, random_split, analyze_split_leakage
from src.models.train import train_all_models, select_best_model, save_best_model
from src.models.evaluate import (
    evaluate_model, plot_all_models_comparison, threshold_analysis, save_comparison_table
)


def main():
    print("====================================================================")
    print("  EXPLAINABLE AI PHISHING URL DETECTION — TRAINING PIPELINE")
    print("====================================================================")

    start_time = time.time()
    config = load_config()
    artifacts_dir = PROJECT_ROOT / config.get("paths", {}).get("artifacts", "artifacts")
    data_raw_dir = PROJECT_ROOT / config.get("paths", {}).get("data_raw", "data/raw")
    data_proc_dir = PROJECT_ROOT / config.get("paths", {}).get("data_processed", "data/processed")

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "plots").mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "metrics").mkdir(parents=True, exist_ok=True)
    data_proc_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    raw_csv = data_raw_dir / "phishing_urls.csv"
    if not raw_csv.exists():
        print(f"[DATA] {raw_csv} not found. Running synthetic sample data generator...")
        import subprocess
        subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "make_sample_data.py")], check=True)

    print(f"\n[STEP 1] Loading raw dataset from {raw_csv}...")
    df = load_dataset(config)
    print(f"Dataset shape: {df.shape}")
    print("Class distribution:")
    print(df["label"].value_counts(normalize=True).rename({0: "Legitimate (0)", 1: "Phishing (1)"}))

    # 2. Extract Features
    print("\n[STEP 2] Extracting lexical and structural features from URL strings...")
    t0_feat = time.time()
    feature_df = batch_extract_features(df["url"].tolist())
    feat_time = time.time() - t0_feat
    feature_names = get_feature_names()
    print(f"Extracted {len(feature_names)} features for {len(df)} URLs in {feat_time:.1f}s.")

    # Save feature schema
    schema_path = artifacts_dir / "feature_schema.json"
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump({"feature_names": feature_names, "feature_count": len(feature_names)}, f, indent=2)
    print(f"Feature schema saved to {schema_path}")

    # Combine features and metadata
    full_df = pd.concat([df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)

    # 3. Data Splitting: Grouped by Registered Domain
    print("\n[STEP 3] Performing leak-free train/val/test splitting grouped by registered domain...")
    X_train_df, X_val_df, X_test_df, y_train_s, y_val_s, y_test_s = group_aware_split(
        full_df, config=config, url_col="url", label_col="label"
    )

    leakage_check = analyze_split_leakage(X_train_df, X_val_df, X_test_df, group_col="registered_domain")
    print(f"Leakage check: {leakage_check}")

    # Also compute random split for comparison
    print("Computing random split comparison to assess domain data leakage...")
    X_tr_r, X_val_r, X_te_r, y_tr_r, y_val_r, y_te_r = random_split(
        full_df, config=config, url_col="url", label_col="label"
    )

    # Convert to feature matrices
    X_train = X_train_df[feature_names].values
    X_val = X_val_df[feature_names].values
    X_test = X_test_df[feature_names].values
    y_train = y_train_s.values
    y_val = y_val_s.values
    y_test = y_test_s.values

    # Save processed splits
    np.save(data_proc_dir / "X_train.npy", X_train)
    np.save(data_proc_dir / "X_val.npy", X_val)
    np.save(data_proc_dir / "X_test.npy", X_test)
    np.save(data_proc_dir / "y_train.npy", y_train)
    np.save(data_proc_dir / "y_val.npy", y_val)
    np.save(data_proc_dir / "y_test.npy", y_test)
    X_test_df[["url"]].to_csv(data_proc_dir / "test_urls.csv", index=False)

    print(f"Splits: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")

    # 4. Train All 6 Models
    print("\n[STEP 4] Training and tuning 6 classification models...")
    models_res = train_all_models(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        config=config,
        feature_names=feature_names,
        force_retrain=True
    )

    # 5. Evaluate All Models on Domain-Grouped Test Set
    print("\n[STEP 5] Evaluating all models on the held-out domain-grouped test set...")
    eval_results = {}
    for m_name, m_info in models_res.items():
        pipe = m_info["pipeline"]
        ev = evaluate_model(
            pipeline=pipe,
            X_test=X_test,
            y_test=y_test,
            model_name=m_name,
            threshold=config.get("modeling", {}).get("threshold", 0.5),
            artifacts_dir=artifacts_dir
        )
        eval_results[m_name] = ev

    # 6. Save Model Comparison Table and Plots
    print("\n[STEP 6] Saving comparison tables and generating evaluation plots...")
    comp_df = save_comparison_table(eval_results, artifacts_dir)
    plot_all_models_comparison(eval_results, y_test, artifacts_dir)

    # 7. Select & Save Champion Model
    best_model_name = select_best_model(models_res, metric="val_roc_auc")
    save_best_model(models_res, best_model_name, config)
    champion_pipe = models_res[best_model_name]["pipeline"]

    # 8. Decision Threshold Analysis for Champion Model
    print(f"\n[STEP 7] Performing decision threshold analysis for champion model '{best_model_name}'...")
    th_info = threshold_analysis(champion_pipe, X_test, y_test, best_model_name, artifacts_dir)
    print(f"Optimal threshold: {th_info['optimal_threshold']:.2f} (F1: {th_info['optimal_f1']:.4f}, Recall: {th_info['optimal_recall']:.4f})")

    # 9. Data Leakage Comparison: Train Champion Model on Random Split
    print("\n[STEP 8] Quantifying domain-leakage effect (Grouped Split vs. Random Split)...")
    X_tr_r_mat = X_tr_r[feature_names].values
    X_te_r_mat = X_te_r[feature_names].values
    y_tr_r_vec = y_tr_r.values
    y_te_r_vec = y_te_r.values

    from copy import deepcopy
    rand_pipe = deepcopy(champion_pipe)
    rand_pipe.fit(X_tr_r_mat, y_tr_r_vec)
    r_probs = rand_pipe.predict_proba(X_te_r_mat)[:, 1]
    r_auc = float(roc_auc_score(y_te_r_vec, r_probs))
    g_auc = eval_results[best_model_name]["roc_auc"]

    leakage_report = {
        "model": best_model_name,
        "grouped_split_roc_auc": round(g_auc, 4),
        "random_split_roc_auc": round(r_auc, 4),
        "leakage_inflation": round(r_auc - g_auc, 4),
        "explanation": "Random split allows URLs from the same registered domain to appear in both train and test partitions, artificially inflating test scores. Domain-grouped splitting eliminates this shortcut, providing an honest measure of generalization to new unseen domains."
    }
    with open(artifacts_dir / "metrics" / "leakage_comparison.json", "w", encoding="utf-8") as f:
        json.dump(leakage_report, f, indent=2)
    print(f"Leakage comparison: Grouped ROC-AUC = {g_auc:.4f} vs. Random ROC-AUC = {r_auc:.4f} (Inflation: +{r_auc - g_auc:.4f})")

    total_time = time.time() - start_time
    print(f"\nTraining pipeline completed in {total_time / 60.0:.2f} minutes.")
    print(f"Best model: {best_model_name.upper()} with ROC-AUC = {g_auc:.4f} on leak-free domain-grouped test split.")


if __name__ == "__main__":
    main()
