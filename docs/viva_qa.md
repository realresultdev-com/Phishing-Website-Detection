# Viva Q&A: Phishing URL Detection with Explainable AI

40 questions and answers covering core ML concepts, XAI theory, and project-specific design decisions.

---

## Part 1: Machine Learning Fundamentals

**Q1. What is the bias-variance tradeoff and why does it matter here?**

Bias is the error from overly simplistic assumptions (underfitting), while variance is the error from excessive sensitivity to training data (overfitting). A high-bias model (e.g., logistic regression with no feature engineering) may miss complex phishing patterns, while a high-variance model (e.g., unpruned decision tree) memorises training URLs and fails on novel attacks. Ensemble methods like Random Forest and XGBoost manage this tradeoff by averaging many high-variance weak learners. Choosing the right model family and regularisation strength is therefore critical for generalising to unseen phishing URLs.

---

**Q2. How do you detect and prevent overfitting in this project?**

Overfitting is detected by comparing validation-set metrics to training-set metrics: a large gap (e.g., train F1 = 0.99, val F1 = 0.80) signals overfitting. Prevention strategies used here include 5-fold cross-validation during hyperparameter search, L1/L2 regularisation for logistic regression, `max_depth` and `min_child_weight` constraints for tree models, and early stopping on the validation set for XGBoost and LightGBM. The 15 % held-out test set is never used during tuning, providing an unbiased final estimate.

---

**Q3. Why use stratified train/test splitting?**

Stratified splitting ensures that the class ratio (phishing vs. legitimate) is preserved in every subset, which is especially important when classes are imbalanced. Without stratification, random chance could place nearly all phishing examples in the training set, causing validation metrics to be unrepresentative. Scikit-learn's `train_test_split(stratify=y)` guarantees that each fold reflects the overall class distribution, giving reliable estimates of recall and F1.

---

**Q4. What is cross-validation and why use 5 folds?**

K-fold cross-validation divides the training data into K equal parts; the model is trained K times, each time using K-1 folds for training and the remaining fold for validation, and the K scores are averaged. Five folds is a practical choice that balances computational cost (5× training passes) against statistical reliability of the performance estimate. With ~165k training examples (70 % of ~235k), each fold has ~33k samples—large enough to give stable metric estimates. More folds (e.g., 10) would increase runtime with marginal statistical gain.

---

**Q5. What is the Matthews Correlation Coefficient (MCC) and why report it?**

MCC is a balanced metric for binary classification that uses all four cells of the confusion matrix (TP, TN, FP, FN): `MCC = (TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))`. It returns a value in [−1, +1], where +1 is perfect prediction, 0 is no better than random, and −1 is total disagreement. Unlike accuracy, MCC is informative even under class imbalance. For phishing detection, where both false positives (blocking legitimate sites) and false negatives (missing attacks) are costly, MCC provides a single balanced summary statistic.

---

**Q6. Why prefer F1 score over accuracy for imbalanced datasets?**

If the dataset is 90 % legitimate, a classifier that always predicts "legitimate" achieves 90 % accuracy while catching zero phishing URLs. F1 is the harmonic mean of precision and recall, penalising models that sacrifice recall (missing phishing) for precision, or vice versa. The harmonic mean is more sensitive to low values than the arithmetic mean, so a model with 0 recall always gets F1 = 0 regardless of precision. For this reason, F1 (and PR-AUC) are primary metrics here.

---

**Q7. What is PR-AUC and when does it outperform ROC-AUC?**

The Precision-Recall AUC summarises the tradeoff between precision and recall across all decision thresholds. ROC-AUC can appear high (~0.90) even when a model performs poorly on the minority class, because the large number of true negatives inflates the ROC curve. PR-AUC focuses solely on the positive (phishing) class and is more informative under severe class imbalance. When the cost of missing phishing (FN) is high, optimising for PR-AUC aligns the model selection criterion with operational requirements.

---

**Q8. How does SMOTE address class imbalance?**

SMOTE (Synthetic Minority Over-sampling Technique) creates synthetic minority-class samples by interpolating between existing minority instances in feature space. For each minority sample, SMOTE selects k nearest neighbours and generates new points along the line segments connecting them. Unlike simple oversampling (duplicating existing samples), SMOTE introduces novel but plausible examples, reducing overfitting to specific minority instances. It is applied only to the training set, never to validation or test data, to avoid information leakage.

---

**Q9. Explain the concept of data leakage and how you prevent it.**

Data leakage occurs when information from outside the training distribution (e.g., from the test set or future observations) is inadvertently used to train or evaluate the model, leading to overly optimistic estimates. In this project, leakage is prevented by: (1) computing feature statistics (e.g., imputation values, scalers) only on the training fold within cross-validation; (2) keeping the test set entirely held out until final evaluation; and (3) never using the target label to construct features. The pipeline uses scikit-learn `Pipeline` objects to encapsulate preprocessing, ensuring transformations are fit only on training data.

---

**Q10. Why might GroupShuffleSplit be preferred over random splitting in some phishing datasets?**

Some phishing datasets contain multiple URLs from the same domain or campaign. Random splitting could place related URLs in both train and test sets, allowing the model to "recognise" a domain it has already seen and inflating test metrics. `GroupShuffleSplit` with the registered domain as the group key ensures that all URLs from the same domain land in either train or test, not both. This gives a stricter, more realistic evaluation of the model's ability to generalise to entirely new domains.

---

## Part 2: SHAP Theory

**Q11. What is SHAP and what is its theoretical foundation?**

SHAP (SHapley Additive exPlanations) assigns each feature an importance value for a specific prediction by computing Shapley values from cooperative game theory. A Shapley value for feature i is the average marginal contribution of feature i across all possible orderings (coalitions) of features. The key axioms satisfied are efficiency (SHAP values sum to the prediction minus the base rate), symmetry (features with identical contributions get equal values), dummy (features that never change the output get zero), and linearity (additivity). SHAP is model-agnostic but specialised, faster variants exist for tree models (`TreeExplainer`) and linear models.

---

**Q12. What is TreeSHAP and why is it preferred for XGBoost/LightGBM?**

`TreeSHAP` is a polynomial-time algorithm that computes exact Shapley values for tree-based models by exploiting the tree structure. Whereas the naive Shapley computation requires exponential time (2^n evaluations for n features), TreeSHAP runs in O(TLD²) time where T is the number of trees, L the maximum number of leaves, and D the maximum depth. This makes it feasible for forests with hundreds of trees and thousands of samples. For XGBoost and LightGBM, `shap.TreeExplainer` automatically uses this algorithm, giving exact (not approximated) Shapley values.

---

**Q13. How do you interpret a SHAP waterfall plot?**

A SHAP waterfall plot for a single prediction shows the base value (average model output over the training set) at the bottom and the final prediction at the top. Each feature's SHAP value is a horizontal bar: red bars indicate features that push the prediction toward phishing, blue bars push toward legitimate. The bars are ordered by magnitude so the most influential features appear first. The cumulative sum of all SHAP values plus the base value equals the model's raw output (log-odds for tree classifiers), which is then converted to a probability.

---

**Q14. What is a SHAP beeswarm (summary) plot?**

A SHAP beeswarm plot shows the distribution of SHAP values for every feature across all samples in the dataset. Each point represents one sample; its x-position is its SHAP value for that feature, and its colour encodes the feature's raw value (red = high, blue = low). Features are ordered by mean absolute SHAP value (global importance) from top to bottom. This allows simultaneous inspection of global feature importance and the direction of each feature's effect, revealing e.g. that "has_IP_address=1" always increases phishing risk.

---

**Q15. How do global and local SHAP explanations differ?**

Global SHAP explanations aggregate individual Shapley values across the entire dataset to summarise which features matter most on average (e.g., mean |SHAP|). Local SHAP explanations are per-instance: they show why the model assigned a specific probability to one specific URL. Global explanations support model understanding and bias auditing; local explanations support user-facing alerts ("This URL was flagged because it uses an IP address and contains the keyword 'login'"). Both are reported in this project.

---

## Part 3: LIME Theory

**Q16. What is LIME and how does it differ from SHAP?**

LIME (Local Interpretable Model-agnostic Explanations) explains a single prediction by training a simple, interpretable surrogate model (usually ridge regression) on a locally perturbed neighbourhood around the input. It perturbs the input, queries the black-box model for predictions on the perturbed samples, weights samples by their proximity to the original input, and fits the surrogate. Unlike SHAP, LIME does not satisfy the Shapley axioms and its explanations can be inconsistent across runs (depending on random perturbations). However, LIME is model-agnostic and intuitive, complementing SHAP's theoretical guarantees.

---

**Q17. What is the fidelity of a LIME explanation?**

Fidelity measures how faithfully the surrogate model (ridge regression) approximates the black-box model's predictions within the local neighbourhood. It is typically computed as 1 − RMSE(surrogate_predictions, black_box_predictions) over the perturbed samples used to fit the surrogate. High fidelity means the surrogate closely mirrors the black-box in the local region, making the explanation trustworthy. Low fidelity indicates the surrogate is a poor local approximation, and the explanation's feature weights should be treated with caution.

---

**Q18. What is LIME explanation stability and why does it matter?**

Stability (also called robustness) measures whether LIME produces consistent explanations for the same input across multiple runs or small input perturbations. It is quantified here as the Jaccard similarity of the top-k features identified across repeated LIME calls on the same URL. Poor stability means a user who runs the explanation twice may see different "reasons", undermining trust in the system. Stability can be improved by increasing `num_samples` (more perturbations) and fixing the random seed, at the cost of higher computation.

---

**Q19. Why use both SHAP and LIME in the same system?**

SHAP and LIME provide complementary guarantees. SHAP offers exact, theoretically grounded Shapley values with consistency and efficiency axioms; LIME is fast, fully model-agnostic, and familiar to practitioners. Using both allows cross-validation: if both methods rank the same features highly for a given URL, the explanation is more credible. Where they disagree, it signals that the local decision boundary is complex and warrants further human review. Dual explanations also serve different audiences (data scientists prefer SHAP; non-technical stakeholders prefer LIME's textual output).

---

**Q20. How is completeness of an explanation evaluated?**

Completeness checks whether the top-k features identified by the explainer account for a sufficient fraction of the total predicted risk. It is measured as: `completeness = |sum of top-k SHAP values| / |total prediction − base value|`. A completeness of 0.90 means the top-k features explain 90 % of the model's output for that prediction. Low completeness indicates that many small-contribution features collectively drive the prediction, making it difficult to summarise compactly.

---

## Part 4: URL Feature Engineering

**Q21. Why use URL-only features instead of page content?**

URL-based detection is fast (no HTTP request required), privacy-preserving (no need to visit potentially malicious content), and applicable to link scanning before the user clicks. Page-content approaches require fetching the page, exposing the scanner to drive-by-download attacks and incurring latency. URL features can be extracted in microseconds, making them suitable for real-time inline proxies and email gateways. The tradeoff is reduced signal: a phishing page on a legitimate CDN with a benign-looking URL can evade purely URL-based systems.

---

**Q22. What are the most informative URL features for phishing detection?**

Empirically, the most discriminative URL features include: (1) presence of an IP address instead of a domain name; (2) URL length (phishing URLs tend to be longer to obscure malicious intent); (3) number of dots and hyphens in the domain; (4) suspicious keywords (login, secure, verify, update, bank); (5) TLD type (phishing frequently uses free TLDs like .tk, .ml, .ga); (6) subdomain depth; and (7) presence of URL shorteners. SHAP global summary plots on the trained model will confirm which features have the highest mean absolute Shapley values on this dataset.

---

**Q23. What is Shannon entropy and how is it used as a URL feature?**

Shannon entropy of a string s is `H = − Σ p_i log₂(p_i)` where p_i is the probability of character i. High-entropy domain names (e.g., `a3f9x2k.com`) are characteristic of algorithmically generated phishing domains (DGA), while legitimate domains have lower entropy due to meaningful words. Entropy is computed separately for the hostname, path, and full URL and included as three numeric features. It is a simple but effective signal for detecting randomly generated subdomains.

---

**Q24. What are the limitations of URL-only features?**

The main limitations are: (1) adversarial URLs crafted to mimic legitimate patterns (short, low entropy, well-known TLD) can evade detection; (2) URL shorteners (bit.ly, tinyurl) hide the true destination, creating high false-negative risk; (3) legitimate services sometimes use long, complex URLs (tracking parameters, OAuth tokens) that may trigger false positives; (4) internationalised domain names (IDN / punycode homograph attacks) require special handling; and (5) the feature set is static and must be updated as phishing tactics evolve.

---

## Part 5: Model Selection

**Q25. Compare XGBoost and LightGBM. When would you prefer one over the other?**

Both are gradient-boosted tree frameworks, but they differ in tree construction strategy. XGBoost uses level-wise (breadth-first) growth, evaluating all possible splits at each level, which can be more accurate but slower. LightGBM uses leaf-wise (best-first) growth, splitting the leaf with the highest loss reduction, achieving faster training with comparable or better accuracy on large datasets. LightGBM also uses histogram-based binning to speed up split finding and uses less memory. For the ~165k-sample training set here, LightGBM is typically 3–10× faster. XGBoost may generalise slightly better on smaller datasets where leaf-wise growth overfits.

---

**Q26. Why include Logistic Regression as a baseline?**

Logistic Regression provides a linear baseline that is fast, interpretable, and well-understood. If the engineered features are good, even a linear model should achieve reasonable performance (F1 > 0.90). A large gap between logistic regression and tree ensembles indicates that feature interactions are important and the ensemble is capturing genuine non-linear patterns, not just memorising noise. Additionally, logistic regression's coefficients serve as an independent linear explanation that can be compared to SHAP values for sanity checking.

---

**Q27. What hyperparameters are most important to tune for XGBoost?**

The most impactful XGBoost hyperparameters are: `n_estimators` (number of trees, with early stopping); `max_depth` (controls tree complexity and overfitting); `learning_rate` (shrinkage; lower = better generalisation but needs more trees); `subsample` and `colsample_bytree` (stochastic sampling, reduces variance); `min_child_weight` (minimum sum of instance weight in a leaf, controls overfitting); and `scale_pos_weight` (class imbalance compensation). `RandomizedSearchCV` over these parameters with 5-fold CV finds a good configuration without exhaustive grid search.

---

## Part 6: Operational & Ethical Considerations

**Q28. What is concept drift and how would you monitor for it?**

Concept drift occurs when the statistical relationship between URL features and the phishing label changes over time, e.g., attackers switch from IP-based URLs to punycode homograph attacks. The model's real-world performance degrades while its training-data metrics remain unchanged. Monitoring strategies include: tracking prediction confidence distributions over time (a shift in average probability indicates drift); scheduled evaluation on labelled batches of recent URLs; and population stability index (PSI) on feature distributions. When drift is detected, the model should be retrained on a fresh, recent dataset.

---

**Q29. What are the ethical considerations of deploying a phishing detector?**

Key ethical concerns are: (1) **false positives** – incorrectly blocking legitimate URLs harms users and businesses, particularly for under-represented TLDs or non-English domains; (2) **transparency** – users should know why a URL was flagged (addressed by SHAP/LIME explanations); (3) **dual-use** – published feature importance could guide adversarial URL crafting; (4) **data privacy** – URL logs may contain sensitive user information and must be handled per data protection regulations; (5) **accountability** – automated systems making security decisions must have clear human oversight and appeal mechanisms.

---

**Q30. What is the cost asymmetry between false positives and false negatives in phishing detection?**

A false negative (missed phishing URL) can result in credential theft, financial fraud, malware installation, or ransomware deployment—high-severity, potentially irreversible harms. A false positive (legitimate URL incorrectly blocked) causes user friction and loss of productivity but is usually low-severity and reversible. This asymmetry justifies lowering the decision threshold below 0.5, accepting more false positives to reduce false negatives, and using recall and PR-AUC as primary optimisation targets rather than accuracy. The optimal threshold depends on the operational context (e.g., consumer vs. enterprise, real-time vs. asynchronous scanning).

---

**Q31. How would you handle adversarial URL evasion?**

Adversarial evasion can be addressed by: (1) adversarial training—adding adversarially crafted URLs to the training set; (2) ensemble diversity—using multiple models so an attacker must simultaneously evade all of them; (3) feature obfuscation—avoiding publishing the full feature schema publicly; (4) adding non-URL signals (certificate age, DNS TTL, WHOIS registration date) that are harder to spoof; and (5) active learning—continuously labelling new suspicious URLs from production traffic to keep the model updated. No purely URL-based system is fully robust; defence-in-depth with other security layers is essential.

---

**Q32. What is dataset bias and how might it affect this project?**

The PhiUSIIL dataset was collected at a specific point in time and may over-represent phishing URLs from popular campaigns of that period. If certain TLDs, brands, or URL patterns are overrepresented, the model may learn to associate those specific patterns with phishing rather than the underlying structural properties. This can cause high test-set accuracy but poor generalisation to real-world traffic. Bias auditing—evaluating performance disaggregated by TLD, URL length bin, and keyword presence—helps identify where the model is reliable and where it is not.

---

## Part 7: Technical Implementation

**Q33. How does the FastAPI endpoint handle concurrent requests?**

FastAPI is built on Starlette (ASGI) and runs on `uvicorn`, an ASGI server. By default, `uvicorn` processes requests asynchronously using Python's `asyncio` event loop. Because feature extraction and model inference are CPU-bound (not I/O-bound), truly concurrent predictions require multiple worker processes (`uvicorn --workers N`). For high-throughput scenarios, the predictor can be wrapped in a background thread pool via `asyncio.run_in_executor` to avoid blocking the event loop. The model is loaded once at startup and shared across all requests via a module-level singleton.

---

**Q34. How is the trained model serialised and loaded?**

The model is serialised using `joblib.dump(model, "artifacts/model.joblib")`, which efficiently handles numpy arrays and scikit-learn estimators via pickle-based serialisation. At serving time, `joblib.load("artifacts/model.joblib")` reconstructs the exact model object. A companion `artifacts/feature_schema.json` records the feature names and expected dtypes, ensuring that the feature extraction pipeline at inference time produces columns in the same order and type as during training. This is a common source of silent bugs if not explicitly managed.

---

**Q35. How is the Streamlit app structured?**

The Streamlit app uses a sidebar for configuration (threshold slider, model version display) and a main panel with three tabs: (1) **Single URL** – paste a URL, see the prediction gauge, SHAP waterfall chart, LIME bar chart, and human-readable reasons; (2) **Batch Analysis** – upload a CSV of URLs and download a results CSV; (3) **Global Insights** – display the SHAP beeswarm and feature importance bar chart from the last training run. Session state caches the loaded model so it is not reloaded on every rerun.

---

**Q36. How are SHAP plots generated and saved?**

SHAP plots are generated using `shap.plots.waterfall`, `shap.plots.beeswarm`, and `shap.plots.bar` from the `shap` library. Because Streamlit does not natively display `shap` matplotlib figures, plots are rendered to a `BytesIO` buffer via `matplotlib.pyplot.savefig(buf, format='png')` and then displayed with `st.image(buf)`. For the report and static exports, figures are saved to `artifacts/plots/` using `plt.savefig("artifacts/plots/shap_summary.png", dpi=150, bbox_inches='tight')`.

---

**Q37. How do you ensure reproducibility across runs?**

Reproducibility is enforced by: (1) setting `random_state=config["project"]["seed"]` on all stochastic components (train/test split, model training, SHAP/LIME sampling); (2) recording `numpy.random.seed` and `random.seed` at script entry points; (3) pinning exact dependency versions in `requirements.txt`; (4) logging the Python, library, and OS versions to a `run_metadata.json` artefact; and (5) version-controlling `config.yaml` so the exact configuration is reproducible. Docker provides environment-level reproducibility for deployment.

---

**Q38. What testing strategy is used for this project?**

The test suite uses `pytest` with four test modules: `test_url_features.py` (unit tests for each feature function with known inputs/outputs), `test_predict.py` (integration test: given a known URL, assert expected label and probability range), `test_api.py` (end-to-end: start the FastAPI test client and assert `/predict` and `/health` responses), and `test_reasons.py` (assert that reason strings are non-empty and contain expected keywords). Coverage is tracked with `pytest-cov`; a CI gate of 80 % line coverage is recommended.

---

**Q39. How would you scale this system to handle 1,000 URL/second?**

Scaling options: (1) **Horizontal scaling** – deploy multiple API replicas behind a load balancer (e.g., nginx or Kubernetes Service); (2) **Batch inference** – batch URL lists and run vectorised feature extraction + model inference on the whole batch rather than one at a time; (3) **Model quantisation / ONNX export** – convert the model to ONNX format for faster CPU inference without Python overhead; (4) **Caching** – cache predictions for recently seen URLs using an in-memory store (Redis) with a short TTL; (5) **Async workers** – use `uvicorn --workers 4` to fully utilise multi-core CPUs for CPU-bound inference.

---

**Q40. What future improvements would most increase real-world utility?**

The highest-impact improvements are: (1) **Multi-modal features** – add certificate validity, WHOIS registration age, and DNS record features to close the URL-only blind spots; (2) **Online learning** – implement a streaming update mechanism so the model continuously adapts to new phishing campaigns without full retraining; (3) **Active learning** – prioritise labelling of uncertain predictions from production traffic to maximise data efficiency; (4) **LLM-enhanced explanations** – use a language model to convert raw SHAP values into more natural, context-aware alert messages; and (5) **Homograph / IDN detection** – add punycode decoding and visual similarity checks to catch IDN-based brand impersonation.
