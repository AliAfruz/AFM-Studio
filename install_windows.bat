@echo off
setlocal
cd /d "%~dp0"
echo AFRUZ AFM STUDIO v0.4.0 - ORBIT WORKSPACE
echo.
if exist ".venv\Scripts\python.exe" goto install
where python >nul 2>&1
if not errorlevel 1 (
    python -m venv .venv
) else (
    py -3 -m venv .venv
)
if not exist ".venv\Scripts\python.exe" (
    echo Python 3.10+ is required. Install Python, then run this installer again.
    pause
    exit /b 1
)
:install
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install --index-url https://mirror2.chabokan.net/pypi/simple/ -r requirements.txt
if errorlevel 1 ".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Dependency installation failed. See the error above.
    pause
    exit /b 1
)
echo.
echo Installation complete. Start with run_windows.bat.
echo OpenGL 3D works independently of optional CUDA filtering.
pause
