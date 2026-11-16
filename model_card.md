# Model Card: Explainable AI Phishing URL Classifier

## Model Details
- **Model Name:** Phishing URL Neural Classifier (MLP Pipeline)
- **Model Version:** 1.0.0
- **Model Type:** Multi-Layer Perceptron (`MLPClassifier`) with `StandardScaler` feature normalization
- **Architecture:** Feed-forward neural network with 2 hidden layers (64, 32 hidden units), ReLU activations, L2 weight decay ($\alpha = 0.001$), Adam optimizer
- **Release Date:** November 2026
- **License:** MIT License
- **Framework:** scikit-learn, joblib, SHAP, LIME

---

## Intended Use
- **Primary Use Case:** Real-time triage and risk assessment of suspicious URLs within Security Operations Centers (SOCs), email gateway filters, and interactive threat hunting tools.
- **Explainability Focus:** Provides local transparent justifications (plain-English risk and protective factor breakdowns) for human analysts before blacklisting or quarantine actions.
- **Out-of-Scope Use Cases:** Not intended as a sole, authoritative perimeter firewall without second-tier verification; does not inspect page content, SSL certificate metadata, or server responses.

---

## Training and Evaluation Data
- **Primary Dataset:** **PhiUSIIL Phishing URL Dataset** (UCI Machine Learning Repository, ID: 967).
- **Cohort Size:** 39,999 stratified samples extracted from 235,795 raw instances (57.2% Phishing, 42.8% Legitimate).
- **Leakage-Free Splitting Strategy:** `GroupShuffleSplit` partitioned on the extracted **Registered Domain** (`eTLD+1`).
  - **Train partition:** 27,882 URLs
  - **Validation partition:** 6,660 URLs
  - **Test partition:** 5,457 URLs (Zero domain overlap with training set)
- **Input Features:** 52 URL-only structural and lexical attributes re-derived from raw string representations without network queries.

---

## Quantitative Metrics (Held-Out Domain-Grouped Test Set)

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | **0.9980** (99.80%) | Overall correctness across all test URLs |
| **Precision** | **0.9971** (99.71%) | Fraction of flagged URLs that were genuinely malicious |
| **Recall (Sensitivity)** | **0.9997** (99.97%) | Proportion of phishing URLs successfully captured |
| **F1-Score** | **0.9984** (99.84%) | Harmonic mean of Precision and Recall |
| **ROC-AUC** | **0.9987** | Discrimination threshold invariance score |
| **PR-AUC** | **0.9985** | Area under Precision-Recall curve |
| **Matthews Correlation Coefficient (MCC)** | **0.9957** | Balanced metric robust to class imbalance |
| **False Positive Rate (FPR)** | **0.0049** (0.49%) | Legitimate URLs erroneously flagged |
| **False Negative Rate (FNR)** | **0.0003** (0.03%) | Dangerous phishing URLs missed (critical safety metric) |
| **Median Inference Latency** | **0.85 ms** | Per-URL latency on standard laptop CPU |
| **Model Size** | **364.0 KB** | Joblib serialized pipeline size |
| **Optimal Decision Threshold** | **0.25** | Threshold maximizing recall (1.0000) while maintaining 0.9985 F1 |

---

## 6-Model Benchmark Comparison Table

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | MCC | Latency (ms) | Size (KB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MLP (Champion)** | **0.9980** | **0.9971** | **0.9997** | **0.9984** | **0.9987** | **0.9985** | **0.9957** | **0.85** | **364.0** |
| Random Forest | 0.9974 | 0.9968 | 0.9991 | 0.9980 | 0.9986 | 0.9987 | 0.9945 | 64.17 | 7065.6 |
| LightGBM | 0.9980 | 0.9974 | 0.9994 | 0.9984 | 0.9985 | 0.9984 | 0.9957 | 2.21 | 149.0 |
| XGBoost | 0.9974 | 0.9971 | 0.9988 | 0.9980 | 0.9984 | 0.9984 | 0.9945 | 0.48 | 120.3 |
| Logistic Regression | 0.9958 | 0.9936 | 0.9997 | 0.9966 | 0.9982 | 0.9984 | 0.9910 | 0.63 | 2.9 |
| Decision Tree | 0.9967 | 0.9971 | 0.9977 | 0.9974 | 0.9976 | 0.9983 | 0.9929 | 0.20 | 10.8 |

---

## Data Leakage Control Analysis
When splitting data randomly, URLs sharing the same registered domain appear in both training and test partitions. This allows models to memorize domain identity rather than generalizable structural cues.
- **Grouped Split ROC-AUC:** 0.9987 (Honest evaluation on unseen domains)
- **Random Split ROC-AUC:** 0.9988 (Slight inflation due to domain overlap: 309 overlapping domains)

---

## Ethical & Security Considerations
1. **Zero-Network Policy:** Designed to never send HTTP/DNS packets to target URLs during feature extraction. This eliminates malware drive-by infection risks and prevents adversary honeypot tripping.
2. **Explainability Transparency:** By exposing plain-English reasons with SHAP and LIME values, security analysts can inspect false positives and avoid unjustified domain takedowns.
3. **Dual-Use Risks:** Adversaries could theoretically inspect feature attribution weights to craft adversarial URLs that minimize sensitive keyword counts or obfuscation ratios. Defense-in-depth is strongly advised.

---

## Limitations
- **Clean / Compromised Domains:** Cannot detect phishing attacks hosted on compromised reputable domains (e.g. `docs.google.com/forms`, compromised WordPress installations) using URL features alone.
- **Adversarial URL Reshaping:** Attackers aware of lexical thresholds can shorten URLs, eliminate hyphens, and use benign subpaths to bypass lexical detection.
- **Concept Drift:** Malicious infrastructure shifts across novel top-level domains and dynamic URL shorteners require continuous retraining.

---

## How to Use
```python
from src.models.predict import load_model, predict_url
from src.explain.reasons import generate_reasons

# Load champion model
pipeline = load_model("artifacts/model.joblib")

# Inference on unvisited URL
result = predict_url("http://paypal-verification.xyz/login.php", pipeline)
print(f"Verdict: {result['label']} (Prob: {result['probability']:.2%})")

# Generate plain-English reasons
reasons = generate_reasons(
    url_features=result['features'],
    prediction_label=result['label'],
    probability=result['probability']
)
print(reasons['plain_english'])
```
