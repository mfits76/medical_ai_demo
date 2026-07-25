@echo off
setlocal
cd /d "%~dp0"

if not exist "artifacts\model.pt" (
  echo Model not found. Training once...
  ".venv\Scripts\python.exe" train.py
  if errorlevel 1 (
    echo Training failed.
    pause
    exit /b 1
  )
)

".venv\Scripts\python.exe" ui.py
if errorlevel 1 (
  echo UI exited with an error.
  pause
)
