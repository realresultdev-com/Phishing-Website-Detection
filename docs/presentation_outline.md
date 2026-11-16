# Presentation Outline: Phishing URL Detection with XAI

**Course:** Final Year B.Tech / M.Tech Project Presentation  
**Duration:** ~20–25 minutes + Q&A  
**Slides:** 17  

---

## Slide 1 — Title

**Title:** An Explainable AI-Based Framework for Phishing Website Detection Using URL Features

**Bullets:**
- Presented by: [Team Names], [Department], [Institution]
- Supervisor: [Supervisor Name]
- Date: [Presentation Date]
- GitHub: [Repository Link]

**Speaker Notes:** Greet the panel. State your name and your project title clearly. Briefly mention this is a final-year project combining machine learning, explainability, and cybersecurity. Transition: "Let me start with why phishing is a problem worth solving."

---

## Slide 2 — Problem Statement

**Title:** The Phishing Threat

**Bullets:**
- Phishing is the #1 delivery vector for credential theft and ransomware
- 3.4 billion phishing emails are sent per day (APWG 2023)
- Average enterprise breach cost: USD 4.76 million (IBM, 2023)
- Traditional blacklists react slowly — zero-day phishing URLs evade them
- Need: a fast, URL-only, ML-based, *explainable* detector

**Speaker Notes:** Give the audience a sense of scale. Emphasise the word "explainable" — a black-box that blocks URLs without reasons is not acceptable in production. The explainability requirement motivates the XAI component and distinguishes this project from prior work.

---

## Slide 3 — Motivation & Existing Work

**Title:** Why This Approach?

**Bullets:**
- Existing ML approaches use page content, DNS, WHOIS — slow and invasive
- URL-only detection: no page fetch, no user privacy compromise, sub-millisecond latency
- Prior work (Mohammad et al. 2016, Sahingoz et al. 2019) achieve F1 ≈ 0.95 but offer no explanations
- Regulatory pressure (GDPR Article 22) demands explainable automated decisions
- Gap: combine URL features + modern XAI (SHAP + LIME) in a deployable system

**Speaker Notes:** Cite two or three real papers briefly. The gap you are filling is the XAI + deployability combination, not just accuracy. Transition: "Here are the objectives we set out to achieve."

---

## Slide 4 — Objectives

**Title:** Project Objectives

**Bullets:**
- Extract ~50 handcrafted URL-lexical features (no page content)
- Train and compare 6 ML classifiers: LR, DT, RF, XGBoost, LightGBM, MLP
- Apply SHAP and LIME for per-prediction explanations
- Quantitatively evaluate explanation quality: fidelity, stability, completeness
- Deploy as REST API (FastAPI) and interactive dashboard (Streamlit)

**Speaker Notes:** These five objectives map directly to the five main modules of the codebase. Each has a measurable deliverable. Transition to architecture which shows how they all connect.

---

## Slide 5 — System Architecture

**Title:** End-to-End Pipeline

**[Include the Mermaid architecture diagram from README.md]**

**Bullets:**
- Raw URL → ~50 features → ML model → probability score
- XAI engine runs SHAP + LIME on every prediction
- Two interfaces: REST API for integration, Streamlit for human review
- All configuration centralised in `config.yaml`

**Speaker Notes:** Walk through the diagram left to right. Emphasise that the same feature extraction code runs at training time and at inference time — no train/serve skew. The XAI engine sits between model and interface so both delivery surfaces benefit from explanations.

---

## Slide 6 — Dataset & EDA

**Title:** PhiUSIIL Dataset

**Bullets:**
- UCI ML Repository (ID 967): ~235,000 URLs
- ~54 % phishing, ~46 % legitimate — near-balanced
- Features cover lexical, structural, keyword, and TLD properties
- EDA reveals: phishing URLs are ~2× longer on average
- Heatmap shows top correlated features with the label

**Speaker Notes:** Show one EDA chart (URL length distribution by class or correlation heatmap). Point out that the near-balanced class ratio reduces the need for aggressive oversampling, though SMOTE is still available as an option in `config.yaml`.

---

## Slide 7 — Feature Engineering

**Title:** URL Feature Groups

**Bullets:**
- **Lexical:** length, digit count, special char count, Shannon entropy
- **Structural:** IP-in-URL, subdomain depth, path depth, file extension
- **Keyword:** login, secure, verify, bank, paypal, update (binary flags)
- **TLD:** suspicious TLD (`.tk`, `.ml`, `.ga`), registered domain length
- **Encoding:** percent-encoding count, punycode detection

**Speaker Notes:** Show a table of the top 10 features by SHAP importance (after training). Explain how each group captures a different attack pattern. For example, IP-in-URL is a classic indicator because legitimate businesses use domain names; entropy catches DGA-generated subdomains.

---

## Slide 8 — Modeling Approach

**Title:** Model Training Pipeline

**Bullets:**
- 70 / 15 / 15 stratified train/val/test split (seed = 42)
- 5-fold `RandomizedSearchCV` (30 iterations) on training set
- 6 models benchmarked: LR, Decision Tree, RF, XGBoost, LightGBM, MLP
- Class imbalance handled: `scale_pos_weight` for tree models, SMOTE optional
- Best model saved to `artifacts/model.joblib`

**Speaker Notes:** Briefly explain why RandomizedSearch is used instead of GridSearch (computational cost). Stress that the test set is used exactly once after all hyperparameter decisions are finalised. Mention that early stopping is used for XGBoost and LightGBM on the validation set.

---

## Slide 9 — Results

**Title:** Model Comparison (Held-Out Domain-Grouped Test Set)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | MCC | Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MLP (Champion)** | **0.9980** | **0.9971** | **0.9997** | **0.9984** | **0.9987** | **0.9985** | **0.9957** | **0.85 ms** |
| Random Forest | 0.9974 | 0.9968 | 0.9991 | 0.9980 | 0.9986 | 0.9987 | 0.9945 | 64.17 ms |
| LightGBM | 0.9980 | 0.9974 | 0.9994 | 0.9984 | 0.9985 | 0.9984 | 0.9957 | 2.21 ms |
| XGBoost | 0.9974 | 0.9971 | 0.9988 | 0.9980 | 0.9984 | 0.9984 | 0.9945 | 0.48 ms |
| Logistic Regression | 0.9958 | 0.9936 | 0.9997 | 0.9966 | 0.9982 | 0.9984 | 0.9910 | 0.63 ms |
| Decision Tree | 0.9967 | 0.9971 | 0.9977 | 0.9974 | 0.9976 | 0.9983 | 0.9929 | 0.20 ms |

**Speaker Notes:** Highlight that the Multi-Layer Perceptron (MLP) achieved the highest overall discrimination with an ROC-AUC of 0.9987 and F1 of 0.9984 on the strictly leak-free domain-grouped test set. Emphasize that all six models demonstrated strong performance, but MLP and gradient boosted trees provided the lowest false negative rates (under 0.1%). Point out the sub-millisecond inference speed (0.85 ms), making it suitable for live inline packet or browser proxy filtering.
| LightGBM            | TBD      | TBD       | TBD    | TBD | TBD     | TBD    | TBD |
| MLP                 | TBD      | TBD       | TBD    | TBD | TBD     | TBD    | TBD |

**Bullets:**
- Best model: XGBoost / LightGBM (TBD after training)
- Decision threshold set to 0.5; can be lowered to increase recall
- MCC used as primary metric for imbalanced-data robustness

**Speaker Notes:** Explain why F1, MCC, and PR-AUC are highlighted over accuracy. Acknowledge that "accuracy looks great" for imbalanced tasks even with naive classifiers. The panel may ask about confusion matrices — be ready to show them.

---

## Slide 10 — Explainability: Global (SHAP)

**Title:** What Does the Model Consider Important? (SHAP)

**[Show SHAP beeswarm / bar chart]**

**Bullets:**
- SHAP (SHapley Additive exPlanations) — game-theoretic, axiomatically fair
- Global: mean |SHAP| across all test samples
- Top features: has_ip_address, url_length, suspicious_tld, keyword_login, entropy_host
- Red = high feature value increases phishing risk; blue = decreases risk

**Speaker Notes:** Walk through the top 5 features on the beeswarm plot. Explain that SHAP values sum to the prediction (efficiency axiom). Note TreeSHAP is exact, not an approximation, for tree models. Transition: "SHAP also works at the individual URL level."

---

## Slide 11 — Explainability: Local (LIME)

**Title:** Why Was *This* URL Flagged? (LIME)

**[Show LIME bar chart for one phishing URL]**

**Bullets:**
- LIME trains a local linear surrogate around the specific URL
- Example: `http://192.168.1.1/secure-login/paypal-verify.php`
- Top reasons: IP in URL, keyword "secure", keyword "paypal", long path
- Both SHAP waterfall and LIME bar shown side-by-side in the app

**Speaker Notes:** Use a memorable, illustrative phishing URL as the demo example. Walk the panel through the LIME bar chart interpretation. Mention that LIME is model-agnostic but less stable than SHAP — which motivates the quantitative evaluation on the next slide.

---

## Slide 12 — Explanation Quality Evaluation

**Title:** How Good Are the Explanations?

**Bullets:**
- **Fidelity:** surrogate RMSE vs. black-box — target: > 0.90 R²
- **Stability:** Jaccard(top-k) across 10 perturbations — target: > 0.75
- **Completeness:** top-10 SHAP values / total prediction — target: > 0.85
- Results across 200 sampled URLs: [TBD after training]
- SHAP > LIME on stability; LIME more intuitive for non-experts

**Speaker Notes:** This slide distinguishes the project from papers that only report model accuracy. Quantitative XAI evaluation is a research contribution. Explain each metric briefly. The panel may ask how stability is measured — Jaccard similarity of the top-k feature index sets.

---

## Slide 13 — Error Analysis

**Title:** Where Does the Model Fail?

**Bullets:**
- False negatives: phishing URLs with no suspicious lexical signals (CDN-hosted, short paths)
- False positives: legitimate auth URLs with long query strings (OAuth, tracking)
- Adversarial: URLs crafted to mimic legitimate patterns
- Error analysis disaggregated by TLD, URL length bin, keyword presence

**Speaker Notes:** Show a small table of FN and FP examples with their feature vectors. This demonstrates honest evaluation and understanding of limitations. Acknowledge that URL-only detection has inherent blind spots and what can be done (threshold tuning, supplementary signals).

---

## Slide 14 — Web App Demo

**Title:** Live Streamlit Dashboard

**[Screenshot or live demo]**

**Bullets:**
- Single URL input → prediction gauge (0–100 % phishing risk)
- SHAP waterfall + LIME bar chart side-by-side
- Human-readable reason strings below the charts
- Batch CSV upload → downloadable results with reasons
- Threshold slider in sidebar

**Speaker Notes:** If possible, do a live demo. Enter a benign URL first (e.g., `https://www.wikipedia.org`) to show the low-risk response, then enter an obviously phishing URL. Walk through the explanation panel. Mention the batch mode for SOC workflows.

---

## Slide 15 — API Demo

**Title:** REST API Integration

**[Show Swagger UI screenshot + curl command]**

**Bullets:**
- FastAPI with auto-generated Swagger UI at `/docs`
- `POST /predict` → JSON response with label, probability, reasons
- `GET /health` → liveness check for Kubernetes probes
- Containerised via Docker: `docker build -t phishing-xai . && docker run -p 8000:8000`

**Speaker Notes:** Show the curl command and the JSON response. Mention that the API is stateless and horizontally scalable. The Docker container can be deployed to any cloud provider. The Swagger UI makes it easy for developers to integrate without reading source code.

---

## Slide 16 — Limitations & Future Work

**Title:** Honest Assessment & Next Steps

**Limitations:**
- URL-only: CDN-hosted phishing, URL shorteners can evade detection
- Adversarial evasion by informed attackers
- Dataset snapshot — concept drift over time
- No multilingual / IDN homograph handling

**Future Work:**
- Add certificate, DNS, WHOIS features
- Online / active learning for continuous adaptation
- LLM-enhanced natural-language explanation generation
- Browser extension deployment

**Speaker Notes:** Being candid about limitations impresses the panel. For each limitation, have a mitigation ready. The future work section shows you understand the research space and have a vision beyond this project.

---

## Slide 17 — Conclusion & References

**Title:** Summary & References

**Conclusion bullets:**
- End-to-end XAI pipeline for URL phishing detection — first to combine SHAP + LIME quality metrics
- Best model achieves F1 = [TBD], ROC-AUC = [TBD] on 35k test URLs
- Dual deployment: Streamlit dashboard + FastAPI, Dockerised
- Open-source; reproducible with `make setup && make data && make train`

**Key References:**
1. Mohammad, R.M. et al. (2016). Using ML to identify phishing websites. *Machine Learning*.
2. Sahingoz, O.K. et al. (2019). Machine learning based phishing detection from URLs. *Computers & Security*.
3. Lundberg, S. & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
4. Ribeiro, M.T. et al. (2016). "Why should I trust you?" LIME. *KDD*.
5. PhiUSIIL Dataset — UCI ML Repository (2023).

**Speaker Notes:** Thank the panel. Restate the three key contributions: (1) URL-only fast detection, (2) dual XAI explanations, (3) quantitative explanation evaluation. Invite questions. Keep the GitHub link visible.
