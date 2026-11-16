#!/usr/bin/env bash
# install.sh - Dependency installer for Linux / macOS
set -e

echo "======================================================================"
echo "  Installing Dependencies for Phishing Website Detection"
echo "======================================================================"
echo ""

if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 not found. Please install Python 3.10+."
    exit 1
fi

if [ ! -d "venv" ]; then
    echo "[INFO] Creating virtual environment..."
    python3 -m venv venv
fi

echo "[INFO] Activating virtual environment..."
source venv/bin/activate

echo "[INFO] Upgrading pip..."
python3 -m pip install --upgrade pip

echo "[INFO] Installing requirements..."
pip install -r requirements.txt

echo ""
echo "======================================================================"
echo "  Installation Complete! Run './run.sh' to launch the app."
echo "======================================================================"
