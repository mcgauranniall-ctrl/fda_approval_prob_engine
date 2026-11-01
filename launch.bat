@echo off
rem Launches the FDA Approval Probability Engine UI (Streamlit) in your browser.
rem Double-click this file, or run it from a terminal. Close the window to stop the app.
setlocal
cd /d "%~dp0"
title FDA Approval Engine

rem -- find uv ------------------------------------------------------------------------------------------
set "UV=uv"
where uv >nul 2>&1 || set "UV=%USERPROFILE%\.local\bin\uv.exe"
if not "%UV%"=="uv" if not exist "%UV%" (
    echo [error] uv was not found. Install it from https://docs.astral.sh/uv/ and try again.
    pause
    exit /b 1
)

rem -- first run: create the Python environment -----------------------------------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo Setting up the Python environment ^(first run only^)...
    "%UV%" sync || (echo [error] uv sync failed. & pause & exit /b 1)
)

rem -- configuration check ----------------------------------------------------------------------------
if not exist ".env" (
    echo [warning] No .env file found. Press releases from SEC EDGAR are disabled until you set
    echo           SEC_USER_AGENT="Your Name your@email". Copy .env.example to .env and fill it in.
    echo.
) else (
    findstr /r /c:"^SEC_USER_AGENT=..*" ".env" >nul || (
        echo [warning] SEC_USER_AGENT is empty in .env, so press releases from SEC EDGAR are disabled.
        echo.
    )
)

echo Starting FDA Approval Engine at http://localhost:8501 ...
echo Close this window to stop it.
echo.
"%UV%" run streamlit run app\streamlit_app.py --server.port 8501 --browser.gatherUsageStats false
if errorlevel 1 pause
endlocal
