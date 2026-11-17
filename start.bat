@echo off
title Pakistani English/Roman-Urdu Lecture Intelligence System
cd /d "%~dp0"
echo ======================================================================
echo   Starting Pakistani English/Roman-Urdu Lecture Intelligence System...
echo ======================================================================
echo.

:: Detect Python executable
set "PY_EXE=python"
if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" (
    set "PY_EXE=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
)

echo   Using Python: %PY_EXE%
echo   Local Web UI: http://127.0.0.1:8000
echo   API Docs:     http://127.0.0.1:8000/docs
echo.
echo   Opening browser automatically...
echo   (Keep this window open while using the app. Press Ctrl+C to stop.)
echo ======================================================================
echo.

"%PY_EXE%" run_server.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ======================================================================
    echo   [INFO] Application closed cleanly.
    echo ======================================================================
    echo.
) else (
    echo.
    echo ======================================================================
    echo   [INFO] Server stopped or exited (Exit Code: %ERRORLEVEL%).
    echo ======================================================================
    echo.
)
pause
