@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found. Create it first:
  echo   python -m venv .venv
  echo   .venv\Scripts\pip install -r requirements.txt
  pause
  exit /b 1
)

if not exist "artifacts\model.pt" (
  echo Model not found. Training once...
  ".venv\Scripts\python.exe" train.py
  if errorlevel 1 (
    echo Training failed.
    pause
    exit /b 1
  )
)

echo Starting web UI at http://127.0.0.1:8000
start "" "http://127.0.0.1:8000"
".venv\Scripts\python.exe" -m uvicorn api:app --host 127.0.0.1 --port 8000
if errorlevel 1 (
  echo Server exited with an error.
  pause
)
