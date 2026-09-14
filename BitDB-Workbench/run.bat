@echo off
setlocal enabledelayedexpansion
title BitDB Workbench - Vector Database & RAG Studio

cd /d "%~dp0"

echo ===================================================
echo   BitDB-Workbench: High-Performance Vector Studio
echo ===================================================

if not exist ".venv\Scripts\python.exe" (
    echo [*] Creating virtual environment (.venv)...
    python -m venv .venv
    if errorlevel 1 (
        echo [!] ERROR: Failed to create virtual environment. Ensure Python 3.11+ is installed.
        pause
        exit /b 1
    )
    echo [*] Installing dependencies from requirements.txt...
    .venv\Scripts\python.exe -m pip install --upgrade pip
    .venv\Scripts\python.exe -m pip install -r requirements.txt
)

echo [*] Launching BitDB Workbench...
.venv\Scripts\python.exe workbench.py %*

if errorlevel 1 (
    echo.
    echo Workbench exited with code %errorlevel%.
    pause
)
