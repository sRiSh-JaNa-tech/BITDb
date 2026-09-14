@echo off
setlocal enabledelayedexpansion
title BitDB - Full System & Prototype Reliability Test Runner

cd /d "%~dp0"

echo =========================================================
echo   BitDB: Automated Reliability & Invariants Test Suite
echo =========================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [!] Virtual environment not found. Running run.bat first to set up environment...
    call run.bat
)

.venv\Scripts\python.exe tests\test_prototypes.py %*

echo.
pause
