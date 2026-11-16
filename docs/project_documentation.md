# PhishingGuard Smart Phishing Website Detection Platform

---

### Download the Project

**Google Drive Download Link:**  
[https://drive.google.com/file/d/your-google-drive-link-here/view?usp=sharing](https://github.com/realresultdev-com/Phishing-Website-Detection)

Download ZIP → Extract → Double-click **`run.bat`**

---

### 1. Project Overview
**PhishingGuard** is a machine learning-powered cybersecurity web application that detects phishing and deceptive websites using URL features only. It explains why a website link was flagged or approved using Explainable AI (SHAP & LIME), generating natural plain-English explanations with zero risk of malware or network visits.

| Feature | Description |
| :--- | :--- |
| 🔍 **URL Scanner** | Real-time analysis of any link using 52 structural indicators |
| 🛡 **Safe Offline Sandbox** | URL strings are evaluated without ever visiting or pinging the host |
| 🧠 **Plain-English Reasons** | Explains exactly why a link is safe or dangerous |
| 📊 **Interactive Risk Gauge** | Color-coded risk percentage score with customizable threshold |
| 📁 **Batch URL Inspection** | Upload a CSV file to scan hundreds of links at once |
| 🏆 **6-Model Benchmark** | Compares XGBoost, LightGBM, Random Forest, MLP, etc. |
| 📈 **Telemetry Curves** | ROC curves, PR curves, SHAP beeswarm, and calibration charts |
| 🔌 **REST API Engine** | FastAPI backend with interactive docs at `/docs` |

---

### 2. System Requirements
| Component | Requirement |
| :--- | :--- |
| **OS** | Windows 10 / 11 (64-bit), macOS, or Linux |
| **Python** | v3.10 or higher — [python.org](https://www.python.org/) |
| **Browser** | Google Chrome (recommended) |
| **RAM** | 4 GB minimum (8 GB recommended) |
| **Internet** | Required only for first-time install; works completely offline at runtime |

> [!NOTE]
> **No database is required.** The app comes with the pre-trained champion machine learning model (`artifacts/model.joblib`) included, making it ready out-of-the-box for college presentations and demos.

---

### 3. How to Run (Windows)
1. Download & Extract ZIP from the Drive link above
2. Install Python if not already installed → [python.org](https://www.python.org/) *(check "Add Python to PATH")*
3. Double-click **`run.bat`** in the extracted folder
4. Wait for the terminal window to set up and launch the app
5. Open Chrome → go to **http://localhost:8501**

> [!WARNING]
> Keep the terminal window open while using the app. Closing it stops the local server.

---

### 4. Entering the Application
No sign-in or registration required. Just select a tab to start:

| Tab | Who Uses It |
| :--- | :--- |
| 🔍 **Check a URL** | Analyze any website link, see risk score and plain-English reasons |
| 📁 **Batch Check** | Upload a CSV file of URLs for bulk threat analysis |
| 📊 **Model Performance** | Review 6-model benchmark tables, ROC curves, and SHAP charts |
| ℹ️ **How It Works** | Learn about the machine learning and explainability pipeline |

---

### 5. Feature Guide

#### 5.1 Check a URL
- Type or paste any website link into the **"Enter website link (URL):"** box.
- Or click one of the quick test sample buttons below the box:
  - `[PayPal (Safe)]`, `[Google (Safe)]`, `[Fake PayPal (.xyz)]`, `[Malicious IP (.php)]`
- Click **"Check URL"**.
- View the **Verdict Card**:
  - **🟢 Safe Link:** Verified legitimate website with high confidence.
  - **🔴 Phishing Alert:** Deceptive scam patterns detected.

#### 5.2 Plain-English Explanations
- Below the verdict, read the **Summary Explanation**.
- View two side-by-side breakdown boxes:
  - **Factors that increase phishing risk:** Flags suspicious patterns like `.php` script extensions or misleading subdomains.
  - **Factors that indicate this link is safe:** Highlights trusted attributes like HTTPS encryption and verified official brand domains.

#### 5.3 Interactive Risk Gauge
- Visual speedometer gauge showing the exact risk score from **0% to 100%**.
- Adjust the **Detection Threshold** slider in the sidebar to test custom sensitivity settings.

#### 5.4 LIME Feature Contributions & Full Feature Table
- Interactive bar chart showing local feature attribution weights.
- Expand **"View all 52 extracted URL features"** to see raw metrics (length, entropy, dots, slashes, etc.).

#### 5.5 Batch Check
- Top Tab → **Batch Check**
- Click **Browse files** and upload any CSV file with a `url` column.
- Click **Analyze URLs**.
- View summary metrics: *Total Analyzed, Flagged Phishing, Verified Safe, Speed (URLs/sec)*.
- Click **"Download Results (CSV)"** to save the audited report.

#### 5.6 Model Performance Telemetry
- Top Tab → **Model Performance**
- Compares 6 classifiers: XGBoost, LightGBM, Random Forest, MLP, Logistic Regression, Decision Tree.
- Displays academic evaluation charts: ROC Curves, Precision-Recall Curves, Global SHAP Beeswarm Plot, and Calibration Plot.

#### 5.7 REST API Integration
- Backend API running on port `8000`.
- Open **http://localhost:8000/docs** in your browser for Swagger API documentation.
- Endpoints: `POST /predict`, `POST /explain`, `POST /predict/batch`, `GET /health`.

---

### 6. Technology Stack
Python 3.10+ · Streamlit · FastAPI · Scikit-Learn · XGBoost · LightGBM · SHAP · LIME · Plotly · Pandas · NumPy · Uvicorn · Pytest

---

### 7. Project Structure
```
PhishingGuard/
├── run.bat                     ← Double-click to start (Windows)
├── run.sh                      ← Start script for Linux / macOS
├── run.py                      ← Cross-platform Python launcher
├── install.bat                 ← Standalone installer (Windows)
├── install.sh                  ← Standalone installer (Linux / Mac)
├── requirements.txt            ← Python library dependencies
├── src/
│   ├── app/streamlit_app.py    ← Web dashboard UI
│   ├── api/main.py             ← FastAPI REST service
│   ├── features/url_features.py← 52 lexical/structural feature extractors
│   ├── explain/reasons.py      ← Plain-English explanation generator
│   └── models/predict.py       ← Model inference pipeline
├── artifacts/
│   ├── model.joblib            ← Pre-trained champion model
│   └── plots/                  ← High-res evaluation charts
└── docs/                       ← Full project reports and manuals
```

---

### 8. Troubleshooting
| Problem | Solution |
| :--- | :--- |
| **App doesn't open** | Wait 5 seconds, then open **http://localhost:8501** manually in Chrome |
| **'python' is not recognized** | Install Python from [python.org](https://python.org), check "Add Python to PATH", restart PC |
| **Port already in use** | Restart computer or close previous terminal window, then run `run.bat` again |
| **White screen on load** | Wait 5 seconds and refresh (F5) — Streamlit is initializing in memory |
| **Target URL takes a second** | Normal behavior: the AI calculates SHAP & LIME local attributions on CPU |

---

### 9. Source Code
🔗 **GitHub:** [https://github.com/realresultdev-com/Phishing-Website-Detection.git](https://github.com/realresultdev-com/Phishing-Website-Detection.git)
