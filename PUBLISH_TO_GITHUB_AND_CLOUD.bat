

@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
:: =========================================================================
:: PUBLISH_TO_GITHUB_AND_CLOUD.bat — Deploy 24/7 to Hugging Face Spaces & GitHub
:: =========================================================================
title OmniVoice — Deploy 24/7 to Hugging Face Spaces ^& GitHub Pages

set "PROJECT_ROOT=D:\ALL PROJECT\OJT PROJECT\OmniVoice"
cd /d "%PROJECT_ROOT%"

set "PYTHON_EXE=C:\Users\Duzaier\AppData\Roaming\uv\python\cpython-3.13.15-windows-x86_64-none\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=%PROJECT_ROOT%\.venv\Scripts\python.exe"

"%PYTHON_EXE%" "%PROJECT_ROOT%\scripts\publish_cloud_24_7.py"

echo.
pause
endlocal
