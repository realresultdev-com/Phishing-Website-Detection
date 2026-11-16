#!/usr/bin/env bash
# run.sh - 1-Click Setup and Launch for Linux / macOS
set -e

echo "======================================================================"
echo "  Explainable AI Phishing Website Detection Framework"
echo "======================================================================"
echo ""

# 1. Check Python 3
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 could not be found. Please install Python 3.10+."
    exit 1
fi

echo "[1/5] Checking Python version..."
python3 --version

# 2. Virtual Environment
if [ ! -d "venv" ]; then
    echo "[2/5] Creating Python virtual environment (venv)..."
    python3 -m venv venv
else
    echo "[2/5] Virtual environment found."
fi

# 3. Activate Virtual Environment
echo "[3/5] Activating virtual environment..."
source venv/bin/activate

# 4. Install Dependencies
echo "[4/5] Checking and installing requirements..."
python3 -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo "[OK] Dependencies ready."

# 5. Check Model Artifacts
if [ ! -f "artifacts/model.joblib" ]; then
    echo "[INFO] Model artifacts not found. Training champion model..."
    if [ ! -f "data/raw/phishing_urls.csv" ]; then
        python3 scripts/make_sample_data.py
    fi
    python3 scripts/train.py
fi

# 6. Launch Web Browser & Streamlit
echo ""
echo "======================================================================"
echo "  [5/5] Launching Web Application..."
echo "  Opening in your default browser at: http://localhost:8501"
echo "  (Press Ctrl+C to stop the server)"
echo "======================================================================"
echo ""

# Open default browser after 2 seconds
if command -v xdg-open &> /dev/null; then
    (sleep 2 && xdg-open http://localhost:8501) &
elif command -v open &> /dev/null; then
    (sleep 2 && open http://localhost:8501) &
fi

streamlit run src/app/streamlit_app.py --server.port 8501
