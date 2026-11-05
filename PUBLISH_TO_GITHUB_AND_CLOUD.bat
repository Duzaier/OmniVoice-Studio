@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
:: =========================================================================
:: PUBLISH_TO_GITHUB_AND_CLOUD.bat — Deploy OmniVoice Studio to GitHub & Cloud
:: =========================================================================
title OmniVoice — Publish to GitHub ^& Cloud Environments

set "PROJECT_ROOT=D:\ALL PROJECT\OJT PROJECT\OmniVoice"
cd /d "%PROJECT_ROOT%"

echo.
echo  =====================================================================
echo   🌐 OMNIVOICE STUDIO — DEPLOY LEN GITHUB ^& CLOUD ENVIRONMENTS
echo  =====================================================================
echo.

set "GIT_EXE=D:\Git\cmd\git.exe"
if not exist "%GIT_EXE%" set "GIT_EXE=git"
set "GH_EXE=C:\Program Files\GitHub CLI\gh.exe"

:: 1. Stage & Commit all latest V2.26 Production & Mobile PWA files
echo  [1/4] Dang dong goi ma nguon V2.26 (Desktop + Mobile PWA + Docker + Cloud)...
"%GIT_EXE%" add -A
"%GIT_EXE%" commit -m "Release V2.26: Swiss Minimalist Desktop & Mobile PWA + Fast Video Composer + Multi-Cloud Deploy" >nul 2>&1
echo        [OK] Da commit ban cap nhat moi nhat.

:: 2. Check GitHub CLI Authentication
if exist "%GH_EXE%" (
    echo.
    echo  [2/4] Kiem tra trang thai dang nhap GitHub CLI...
    "%GH_EXE%" auth status >nul 2>&1
    if !ERRORLEVEL! NEQ 0 (
        echo  [INFO] Ban chua dang nhap GitHub CLI. Trinh duyet se mo de ban xac thuc 1 cham...
        "%GH_EXE%" auth login --web -p https
    )
    echo.
    echo  [3/4] Dang tao va day len GitHub Repository (OmniVoice-Studio)...
    "%GH_EXE%" repo create OmniVoice-Studio --public --source=. --remote=github --push
    if !ERRORLEVEL! NEQ 0 (
        "%GIT_EXE%" push -u github HEAD:main --force
    )
    echo.
    echo  [4/4] Hoan tat! Repo GitHub cua ban da san sang de ket noi voi:
    echo        - GitHub Pages (Tu dong chay qua .github/workflows/deploy-pages.yml)
    echo        - Hugging Face Spaces (Import tu GitHub -> Chay app.py)
    echo        - Render.com (Blueprint render.yaml)
    echo        - Railway.app (railway.json / Dockerfile)
) else (
    echo  [WARN] Khong tim thay GitHub CLI tai %GH_EXE%.
)

echo.
pause
endlocal
