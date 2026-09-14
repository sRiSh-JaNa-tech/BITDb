# BitDB Workbench PowerShell Launcher
$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "  BitDB-Workbench: High-Performance Vector Studio" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "[*] Creating virtual environment (.venv)..." -ForegroundColor Yellow
    python -m venv .venv
    Write-Host "[*] Installing dependencies..." -ForegroundColor Yellow
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r requirements.txt
}

Write-Host "[*] Launching BitDB Workbench..." -ForegroundColor Green
& $venvPython workbench.py $args
