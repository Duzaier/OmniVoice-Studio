@echo off
:: =========================================================================
:: OmniVoice Production Workstation Launcher
:: =========================================================================
title OmniVoice Production Workstation
cd /d "%~dp0"

echo.
echo  =============================================================
echo   🎙️ OMNIVOICE PRODUCTION TTS & VOICE CLONING WORKSTATION
echo  =============================================================
echo.
echo  Dang khoi dong he thong...
echo  Dia chi truy cap: http://localhost:8001
echo  Nhan Ctrl+C de dung server.
echo.

set PYTHONIOENCODING=utf-8
set HF_ENDPOINT=https://hf-mirror.com

uv run python omnivoice/cli/workstation.py --port 8001
pause
