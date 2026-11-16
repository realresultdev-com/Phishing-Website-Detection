@echo off
setlocal enabledelayedexpansion
title Install Dependencies - Phishing Website Detection

echo ======================================================================
echo   Installing Dependencies for Phishing Website Detection
echo ======================================================================
echo.

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python not found in system PATH.
    echo Please install Python 3.10+ and check "Add Python to PATH".
    pause
    exit /b 1
)

if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
)

echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

echo [INFO] Installing requirements from requirements.txt...
pip install -r requirements.txt

echo.
echo ======================================================================
echo   Installation Complete!
echo   You can now double-click 'run.bat' to start the web app.
echo ======================================================================
pause
