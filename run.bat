@echo off
setlocal enabledelayedexpansion
title Phishing Website Detection - Explainable AI

echo ======================================================================
echo   Explainable AI Phishing Website Detection Framework
echo ======================================================================
echo.

:: 1. Locate Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found in your system PATH!
    echo Please install Python 3.10 or higher from: https://www.python.org/
    echo Be sure to check the box: "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

:: Display Python version
echo [1/5] Checking Python environment...
python --version

:: 2. Setup Virtual Environment (if not exists)
if not exist "venv" (
    echo.
    echo [2/5] Creating Python virtual environment (venv)...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        echo Please ensure python-venv is available.
        pause
        exit /b 1
    )
) else (
    echo [2/5] Virtual environment found.
)

:: 3. Activate Virtual Environment
echo [3/5] Activating virtual environment...
call venv\Scripts\activate.bat

:: 4. Install Dependencies
echo [4/5] Checking and installing required packages...
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [WARNING] Some dependencies failed silent install. Retrying with full output...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Dependency installation failed.
        pause
        exit /b 1
    )
)
echo [OK] All dependencies installed successfully.

:: 5. Check Model Artifacts (auto-train if first run)
if not exist "artifacts\model.joblib" (
    echo.
    echo [INFO] First-time setup: Training champion machine learning model...
    if not exist "data\raw\phishing_urls.csv" (
        echo [INFO] Generating initial dataset...
        python scripts\make_sample_data.py
    )
    python scripts\train.py
    if %errorlevel% neq 0 (
        echo [ERROR] Model training failed.
        pause
        exit /b 1
    )
)

:: 6. Launch Web Browser & Start Streamlit
echo.
echo ======================================================================
echo   [5/5] Launching Web Application...
echo   Opening in your default browser at: http://localhost:8501
echo   (Press Ctrl+C in this terminal window to stop the server)
echo ======================================================================
echo.

:: Open browser after a 2-second delay in background
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://localhost:8501"

:: Start Streamlit server
streamlit run src\app\streamlit_app.py --server.port 8501

pause
