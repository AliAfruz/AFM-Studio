@echo off
setlocal
cd /d "%~dp0"
set QT_OPENGL=desktop
set PYTHONUTF8=1
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" afm_studio.py
) else (
    python afm_studio.py
)
if errorlevel 1 (
    echo.
    echo If dependencies are missing, run install_windows.bat first.
    pause
)
