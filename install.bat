@echo off
setlocal enabledelayedexpansion

:: Set paths
set "SCRIPT_DIR=%~dp0"
set "PYTHON_VERSION=3.11.9"
set "VENV_DIR=%SCRIPT_DIR%.venv"

echo Setting up environment in: %SCRIPT_DIR%

:: ---------------------------------------------------------------
:: Ensure uv is installed
:: ---------------------------------------------------------------
where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo uv not found, installing uv...
    powershell -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"

    :: uv installs to %USERPROFILE%\.local\bin by default; make it visible in this session
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"

    where uv >nul 2>&1
    if !errorlevel! neq 0 (
        echo Failed to find uv after installation. Please restart this script in a new terminal.
        pause
        exit /b 1
    )
) else (
    echo uv already installed, skipping...
)

:: ---------------------------------------------------------------
:: Let uv manage the Python install (no manual embedded zip / get-pip needed)
:: ---------------------------------------------------------------
echo Ensuring Python %PYTHON_VERSION% is available via uv...
uv python install %PYTHON_VERSION%

:: ---------------------------------------------------------------
:: Create the virtual environment
:: ---------------------------------------------------------------
if exist "%VENV_DIR%\Scripts\python.exe" (
    echo Virtual environment already exists, skipping creation...
) else (
    echo Creating virtual environment...
    uv venv "%VENV_DIR%" --python %PYTHON_VERSION%
)

set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"

echo Installing required packages with uv...

:: Check for NVIDIA GPU
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo NVIDIA GPU detected!
    :: Get CUDA version
    for /f "tokens=3" %%i in ('nvidia-smi ^| findstr "CUDA Version"') do set "CUDA_VERSION=%%i"
    echo Detected CUDA Version: %CUDA_VERSION%

    :: cu126 build: forward-compatible with the machine's CUDA 12.8/13.1 toolkits; onnxruntime-gpu comes from requirements.txt
    echo Installing PyTorch with CUDA 12.6 support...
    uv pip install --python "%VENV_PYTHON%" torch==2.14.0+cu126 torchvision==0.29.0+cu126 --index-url https://download.pytorch.org/whl/cu126
) else (
    echo No NVIDIA GPU detected, installing CPU-only versions...
    uv pip install --python "%VENV_PYTHON%" torch==2.14.0+cpu torchvision==0.29.0+cpu --index-url https://download.pytorch.org/whl/cpu
)

:: Install other requirements from requirements.txt
echo Installing other requirements from requirements.txt...
uv pip install --python "%VENV_PYTHON%" -r "%SCRIPT_DIR%requirements.txt"

:: Re-pin torch/torchvision to the cu126 build: diffusers/transformers/peft depend on an
:: unpinned torch, so the requirements step above upgrades it to the latest PyPI wheel.
:: (This is the "re-pin AFTER this file" mentioned in requirements.txt.)
uv pip install --python "%VENV_PYTHON%" torch==2.14.0+cu126 torchvision==0.29.0+cu126 --index-url https://download.pytorch.org/whl/cu126

:: Create launch script if it doesn't exist
if not exist "%SCRIPT_DIR%\start.bat" (
    echo Creating launch script...
    (
    echo @echo off
    echo cd /d "%%~dp0"
    echo ".venv\Scripts\python.exe" run.py
    echo pause
    ) > "%SCRIPT_DIR%\start.bat"
)

echo Setup complete!
echo Run start.bat to launch the application.
pause