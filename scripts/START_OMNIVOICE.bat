@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
:: =========================================================================
:: START_OMNIVOICE.bat — OmniVoice AI Voice, Video & Mobile Studio (V2.23)
:: Optimized for Dell Inspiron 15 3515 (AMD Ryzen 5 3450U + Radeon Vega 8)
:: =========================================================================
title OmniVoice AI Voice, Video ^& Mobile Studio (V2.23)

set "PROJECT_ROOT=D:\ALL PROJECT\OJT PROJECT\OmniVoice"
if not exist "%PROJECT_ROOT%\omnivoice\cli\workstation.py" (
    set "PROJECT_ROOT=%~dp0"
)
cd /d "%PROJECT_ROOT%"

echo.
echo  =====================================================================
echo   🎙️🎬📱 OMNIVOICE AI VOICE, VIDEO ^& MOBILE STUDIO (V2.23)
echo   Optimized for Dell Inspiron 15 3515 (AMD Ryzen 5 3450U + Vega 8)
echo  =====================================================================
echo.

:: 1. Verify Python virtual environment
set "VENV_PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe"
if not exist "%VENV_PYTHON%" (
    echo  [ERROR] Khong tim thay moi truong Python tai:
    echo          %VENV_PYTHON%
    echo  Vui long kiem tra lai thu muc D:\ALL PROJECT\OJT PROJECT\OmniVoice\.venv
    pause
    exit /b 1
)

:: 2. Protect C: drive (redirect all TEMP / Gradio / Torch caches to D: drive)
if not exist "%PROJECT_ROOT%\cache\temp" mkdir "%PROJECT_ROOT%\cache\temp"
if not exist "%PROJECT_ROOT%\cache\gradio" mkdir "%PROJECT_ROOT%\cache\gradio"
if not exist "%PROJECT_ROOT%\logs" mkdir "%PROJECT_ROOT%\logs"

set "TEMP=%PROJECT_ROOT%\cache\temp"
set "TMP=%PROJECT_ROOT%\cache\temp"
set "TMPDIR=%PROJECT_ROOT%\cache\temp"
set "GRADIO_TEMP_DIR=%PROJECT_ROOT%\cache\gradio"

:: 3. Configure CPU MKL / OpenMP optimal threading (6 threads for Ryzen 5 3450U)
set "OMP_NUM_THREADS=6"
set "MKL_NUM_THREADS=6"
set "PYTHONIOENCODING=utf-8"
set "HF_ENDPOINT=https://hf-mirror.com"

echo  [INFO] Thu muc du an        : %PROJECT_ROOT%
echo  [INFO] Cache / Temp (O D:)  : %TEMP%
echo  [INFO] CPU Threading        : 6 Threads (Intel MKL / oneDNN AVX2)
echo  [INFO] Web Studio (PC/Mobile): http://localhost:8001
echo  [INFO] App Mobile PWA       : http://localhost:8001/mobile
echo.
echo  Dang khoi dong OmniVoice Engine va mo trinh duyet...
echo  (Nhan Ctrl+C de dung server)
echo.

"%VENV_PYTHON%" "%PROJECT_ROOT%\omnivoice\cli\workstation.py" --port 8001 %*
if !ERRORLEVEL! NEQ 0 (
    echo.
    echo  [ERROR] Ung dung ket thuc voi ma loi !ERRORLEVEL!.
    echo  Kiem tra log tai: %PROJECT_ROOT%\logs
    pause
)
endlocal
