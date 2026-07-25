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

if not exist "data\specialty_dictionary.json" (
  echo Missing data\specialty_dictionary.json
  pause
  exit /b 1
)

echo Training from data\specialty_dictionary.json ...
echo.

".venv\Scripts\python.exe" train.py %*
if errorlevel 1 (
  echo.
  echo Training failed.
  pause
  exit /b 1
)

echo.
echo Done. Updated artifacts\ and static\model.json
pause
