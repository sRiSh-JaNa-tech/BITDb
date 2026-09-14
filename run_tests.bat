@echo off
setlocal enabledelayedexpansion
title BitDB - Global Reliability Test Runner

cd /d "%~dp0BitDB-Workbench"

if not exist ".venv\Scripts\python.exe" (
    echo [!] Initializing BitDB-Workbench virtual environment...
    call run.bat
)

.venv\Scripts\python.exe tests\test_prototypes.py %*

echo.
pause
