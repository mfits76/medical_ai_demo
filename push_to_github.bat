@echo off
cd /d "%~dp0"

echo Pushing to https://github.com/mfits76/ai_example
echo.
echo If this fails with "Repository not found", create the repo first:
echo   1. Open https://github.com/new
echo   2. Repository name: ai_example (owner: mfits76)
echo   3. Leave it empty (no README, no .gitignore)
echo   4. Click Create repository
echo   5. Run this script again
echo.

git push -u origin main
pause
