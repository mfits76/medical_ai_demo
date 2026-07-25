@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo Pushing to https://github.com/mfits76/medical_ai_demo
echo.
echo If this fails with "Repository not found", create the repo first:
echo   1. Open https://github.com/new
echo   2. Repository name: medical_ai_demo (owner: mfits76)
echo   3. Leave it empty (no README, no .gitignore)
echo   4. Click Create repository
echo   5. Run this script again
echo.

git add -A
git status --short
echo.

git diff --cached --quiet
if errorlevel 1 (
  set /p COMMIT_MSG=Commit message: 
  if "!COMMIT_MSG!"=="" set "COMMIT_MSG=Update project files."
  git commit -m "!COMMIT_MSG!"
  if errorlevel 1 (
    echo Commit failed.
    pause
    exit /b 1
  )
  echo.
) else (
  echo Nothing new to commit.
  echo.
)

git push -u origin main
if errorlevel 1 (
  echo Push failed.
  pause
  exit /b 1
)

echo.
echo Done.
pause
