# PhishingGuard AI — Explainable AI Phishing Website Detection Platform

### Download the Project
**Google Drive Download Link:**  
[https://drive.google.com/file/d/your-drive-link-here/view?usp=sharing](https://github.com/realresultdev-com/Phishing-Website-Detection) *(or clone from GitHub)*

**Quick Start:** Download ZIP → Extract → Double-click **`run.bat`**

---

## 1. Project Overview

**PhishingGuard AI** is a machine learning-powered cybersecurity web application that detects phishing websites using only raw URL strings—with zero network requests, page scraping, or website visits. Beyond binary classification, it leverages state-of-the-art **Explainable AI (SHAP & LIME)** to generate transparent, plain-English rationales explaining exactly *why* a link was classified as safe or malicious.

| Feature | Description |
| :--- | :--- |
| **Instant URL Scanner** | Real-time analysis of any web address using 52 structural and lexical indicators |
| **Plain-English Explanations** | Translates complex mathematical SHAP/LIME values into clear, human-readable bullet points |
| **Interactive Risk Gauge** | Color-coded risk score with adjustable sensitivity threshold slider |
| **Batch URL Audit** | Upload a CSV file of hundreds of links and export a full threat triage report |
| **Multi-Model Benchmark** | Side-by-side comparison of 6 classifiers (XGBoost, LightGBM, Random Forest, MLP, etc.) |
| **Diagnostic Curves** | Overlaid ROC curves, Precision-Recall curves, SHAP beeswarm plots, and calibration charts |
| **Zero-Network Sandbox** | Offline lexical parsing guarantee—the target URL is never visited or pinged |
| **Production REST API** | FastAPI backend with auto-generated documentation for integration into security pipelines |

---

## 2. System Requirements

| Component | Requirement |
| :--- | :--- |
| **Operating System** | Windows 10 / 11 (64-bit), macOS, or Linux |
| **Python** | Version 3.10 or higher — [python.org](https://www.python.org/) |
| **Web Browser** | Google Chrome, Edge, Firefox, or Safari |
| **RAM** | 4 GB minimum (8 GB recommended) |
| **Storage Space** | ~500 MB free disk space |
| **Internet** | Required only during first-time package setup; completely offline at runtime |

> [!NOTE]
> **No database or remote server setup required.** The application runs fully on local CPU without GPU requirements. Pre-trained champion model artifacts are included, making it ready for live demonstrations.

---

## 3. How to Run (Windows)

1. **Download & Extract** the ZIP file (or clone from GitHub).
2. Ensure **Python 3.10+** is installed → [python.org](https://www.python.org/) *(make sure to check "Add Python to PATH" during installation)*.
3. Double-click **`run.bat`** in the extracted folder.
4. The script will automatically:
   - Create and activate a local virtual environment (`venv`)
   - Install all required libraries from `requirements.txt`
   - Verify model artifacts (auto-trains in ~10 seconds if first run)
   - Open your default web browser automatically to **`http://localhost:8501`**

> [!IMPORTANT]
> **Keep the terminal window open** while using the application. Closing the terminal stops the local server.

---

### How to Run on Linux / macOS

```bash
# 1. Open Terminal and navigate to the project directory
cd Phishing-Website-Detection

# 2. Make the run script executable and run it
chmod +x run.sh
./run.sh
```

*(Or simply run `python run.py` on any operating system.)*

---

## 4. Entering the Application

No login, signup, or API keys are required. The dashboard organizes capabilities into 4 tabs:

| Tab | Who Uses It / What It Does |
| :--- | :--- |
| **Check a URL** | Scan single links, view risk scores, and inspect natural-language SHAP/LIME explanations |
| **Batch Check** | Security analysts auditing a CSV list of domain feeds or suspicious email links |
| **Model Performance** | Review college project benchmarks, ROC/PR curves, and 6-model comparison tables |
| **How It Works** | System architecture, threat modeling, and explainability methodology summary |

---

## 5. Feature Guide

### 5.1 Real-Time URL Scanner
- Navigate to the **Check a URL** tab.
- Choose from preset samples (PayPal, Google, Phishing lookalikes) or paste any custom web link into the text box.
- Click **Check URL**.
- View the **Verdict Card**:
  - **Safe Link (Green):** Verified legitimate structure with high benign confidence.
  - **Phishing Alert (Red):** Deceptive patterns detected (e.g. suspicious subdomains, IP addresses, fake file extensions).

### 5.2 Interactive Risk Gauge
- Displays the continuous threat probability from **0% to 100%**.
- Adjust the **Detection Threshold** slider in the left sidebar (default `0.50`) to observe how changing sensitivity impacts classification decisions.

### 5.3 Plain-English Explanation Engine
- Automatically translates feature attributions into natural sentences:
  - **Factors increasing risk:** Highlighted in soft red (e.g., *"Links directly to a script extension (.php)"*, *"Domain contains sensitive keywords"*).
  - **Factors indicating safety:** Highlighted in soft green (e.g., *"Uses secure encrypted HTTPS protocol"*, *"Domain matches verified official brand (paypal.com)"*).

### 5.4 LIME Local Feature Weights
- Displays a horizontal bar chart showing the top local surrogate weights computed by LIME.
- Green bars represent features supporting a legitimate verdict; red bars indicate features pushing toward phishing.

### 5.5 Full 52-Feature Matrix
- Click the expandable panel **"View all 52 extracted URL features"** to audit every calculated value (entropy, length, character ratios, subdomains, etc.).

### 5.6 Batch Inspection
- Switch to the **Batch Check** tab.
- Click **Browse files** and upload any CSV file containing a column named `url`.
- Click **Analyze URLs**.
- Review the summary cards: *Total Analyzed, Flagged Phishing, Verified Safe, Speed (URLs/sec)*.
- Click **Download Results (CSV)** to export the annotated threat report.

### 5.7 Model Performance & Telemetry
- Switch to the **Model Performance** tab.
- View the academic benchmark table comparing:
  - **XGBoost, LightGBM, Random Forest, Multi-Layer Perceptron (MLP), Logistic Regression, and Decision Tree**.
  - Metrics reported: *Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, MCC, False Positive Rate (FPR), False Negative Rate (FNR), Latency, and Size*.
- View diagnostic plots: Overlaid ROC curves, Precision-Recall curves, SHAP summary beeswarm, and calibration curves.

### 5.8 REST API (FastAPI)
- The backend also runs a production REST API on port `8000`:
  - Interactive Documentation: **`http://localhost:8000/docs`**
  - Endpoints:
    - `POST /predict` — Fast binary verdict and risk probability
    - `POST /explain` — Full prediction + SHAP feature attributions + plain-English reasons
    - `POST /predict/batch` — Bulk URL triage
    - `GET /health` — Service health telemetry

---

## 6. Technology Stack

- **Machine Learning & Data:** Python 3.10+, Scikit-Learn, XGBoost, LightGBM, Pandas, NumPy
- **Explainable AI (XAI):** SHAP (Shapley Additive Explanations), LIME (Local Interpretable Model-agnostic Explanations)
- **Web UI & Visualization:** Streamlit, Plotly Express & Graph Objects
- **Backend API:** FastAPI, Uvicorn, Pydantic
- **URL Lexical Engine:** `urllib.parse`, `tldextract` *(bundled offline cache)*
- **Testing & Packaging:** Pytest, Docker, Makefile

---

## 7. Project Structure

```
Phishing-Website-Detection/
├── run.bat                         # 1-Click launcher for Windows (Install + Run)
├── run.sh                          # 1-Click launcher for Linux / macOS
├── run.py                          # Cross-platform Python launcher
├── install.bat                     # Standalone dependency installer (Windows)
├── install.sh                      # Standalone dependency installer (Linux/Mac)
├── requirements.txt                # Python library dependencies
├── config.yaml                     # Central project configuration
├── README.md                       # Project overview and quickstart
├── model_card.md                   # Model governance and intended use
├── Dockerfile                      # Container deployment specification
├── Makefile                        # CLI build automation
├── data/
│   ├── raw/                        # Raw URL datasets (PhiUSIIL benchmark)
│   └── processed/                  # Leak-free domain-grouped splits (.csv, .npy)
├── artifacts/
│   ├── model.joblib                # Trained champion classification pipeline
│   ├── feature_schema.json         # 52-dimensional feature schema
│   ├── plots/                      # High-resolution ROC, PR, SHAP, and calibration charts
│   └── metrics/                    # Benchmark comparison tables (CSV, JSON, Markdown)
├── notebooks/
│   └── phishing_xai_end_to_end.ipynb # Jupyter demonstration notebook for vivas
├── src/
│   ├── features/url_features.py    # 52 handcrafted lexical/structural feature extractors
│   ├── data/load.py & split.py     # Domain-grouped leak-free partitioners
│   ├── models/train.py & eval.py   # 6-classifier training & evaluation suite
│   ├── explain/                    # SHAP, LIME, and plain-English rationale generators
│   ├── app/streamlit_app.py        # Streamlit web application
│   └── api/main.py                 # FastAPI production endpoints
├── tests/                          # 37 unit and integration tests (Pytest)
└── docs/
    ├── project_guide.md            # Complete user and setup manual
    ├── report.md                   # Full academic capstone thesis report
    ├── architecture.md             # System design and dataflow diagrams
    ├── viva_qa.md                  # Comprehensive viva-voce questions and answers
    └── presentation_outline.md     # Slide-by-slide presentation script
```

---

## 8. Troubleshooting

| Problem | Solution |
| :--- | :--- |
| **Browser doesn't open automatically** | Wait 5 seconds, then manually open Chrome and go to `http://localhost:8501`. |
| **`python` is not recognized** | Install Python from [python.org](https://www.python.org/) and make sure to check **"Add Python to PATH"** during installation, then restart your terminal. |
| **Port 8501 already in use** | Open Task Manager, end any running `python.exe` processes, or run `streamlit run src/app/streamlit_app.py --server.port 8502`. |
| **First run takes a couple of minutes** | The first launch automatically installs libraries and initializes the model. Subsequent launches will start instantly in ~3 seconds. |
| **Target URL shows "Loading..."** | The explainability engine is calculating local SHAP and LIME values. This takes ~1 second per URL on CPU. |
| **Model says "Offline" in sidebar** | Run `python scripts/train.py` or double-click `run.bat` to rebuild the model artifacts. |

---

## 9. Source Code

- **GitHub Repository:** [https://github.com/realresultdev-com/Phishing-Website-Detection.git](https://github.com/realresultdev-com/Phishing-Website-Detection.git)
- **FastAPI Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Streamlit Local Web App:** [http://localhost:8501](http://localhost:8501)
