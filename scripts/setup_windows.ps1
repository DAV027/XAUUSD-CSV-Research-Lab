$ErrorActionPreference = "Stop"
py -3.12 -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,mt5]"
python scripts\verify_environment.py
Write-Host "Environment ready. Keep MT5 open and logged in before exporting XAUUSD." -ForegroundColor Green
