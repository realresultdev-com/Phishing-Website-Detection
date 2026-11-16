"""
run.py
------
Cross-platform 1-click launcher for Explainable AI Phishing URL Detection.
Installs missing dependencies, trains model if needed, opens web browser,
and starts the Streamlit dashboard.

Usage:
    python run.py
"""

import sys
import subprocess
import time
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

def check_and_install_dependencies():
    """Verify core dependencies and install if missing."""
    req_file = PROJECT_ROOT / "requirements.txt"
    if not req_file.exists():
        return

    print("Checking required dependencies...")
    try:
        import streamlit
        import sklearn
        import xgboost
        import lightgbm
        import shap
        import lime
        import plotly
        import tldextract
        print("All core dependencies are present.")
    except ImportError:
        print("Missing dependencies detected. Installing from requirements.txt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_file)])
        print("Dependencies installed successfully.")

def ensure_model_artifacts():
    """Train champion model if artifacts are missing."""
    model_path = PROJECT_ROOT / "artifacts" / "model.joblib"
    if not model_path.exists():
        print("Champion model artifact not found. Initializing training...")
        raw_data = PROJECT_ROOT / "data" / "raw" / "phishing_urls.csv"
        if not raw_data.exists():
            print("Generating initial dataset...")
            subprocess.check_call([sys.executable, str(PROJECT_ROOT / "scripts" / "make_sample_data.py")])
        subprocess.check_call([sys.executable, str(PROJECT_ROOT / "scripts" / "train.py")])
        print("Model training complete.")

def main():
    print("=" * 60)
    print("  Explainable AI Phishing Website Detection Dashboard")
    print("=" * 60)
    print()

    check_and_install_dependencies()
    ensure_model_artifacts()

    url = "http://localhost:8501"
    print()
    print("=" * 60)
    print(f"  Launching Web Application at: {url}")
    print("  Opening default web browser in 2 seconds...")
    print("  (Press Ctrl+C to stop the server)")
    print("=" * 60)
    print()

    # Open browser after slight delay
    def open_browser():
        time.sleep(2)
        webbrowser.open(url)

    import threading
    t = threading.Thread(target=open_browser)
    t.daemon = True
    t.start()

    # Launch Streamlit server
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(PROJECT_ROOT / "src" / "app" / "streamlit_app.py"),
        "--server.port", "8501"
    ]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nServer stopped.")

if __name__ == "__main__":
    main()
