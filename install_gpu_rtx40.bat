@echo off
setlocal
cd /d "%~dp0"
set "AFRUZ_PYTHON=python"
if exist ".venv\Scripts\python.exe" set "AFRUZ_PYTHON=.venv\Scripts\python.exe"
echo ============================================================
echo      AFRUZ AFM STUDIO - OPTIONAL RTX CUDA FILTERS
echo ============================================================
echo.
echo NOTE: Interactive 3D already uses your GPU through OpenGL.
echo This optional package accelerates Gaussian and median filters.
echo.
"%AFRUZ_PYTHON%" -m pip install --upgrade pip
"%AFRUZ_PYTHON%" -m pip install --index-url https://mirror2.chabokan.net/pypi/simple/ -r requirements-gpu.txt
if errorlevel 1 (
    echo.
    echo Chabokan mirror install failed. Trying standard PyPI...
    "%AFRUZ_PYTHON%" -m pip install -r requirements-gpu.txt
)
if errorlevel 1 (
    echo.
    echo CUDA package installation failed.
    echo Make sure the NVIDIA driver is current and your Python version is supported by CuPy.
    pause
    exit /b 1
)
echo.
echo CUDA support installed. Restart Afruz AFM Studio.
pause
