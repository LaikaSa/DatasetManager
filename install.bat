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

    :: Install appropriate PyTorch and ONNX Runtime version based on CUDA
    if "%CUDA_VERSION:~0,4%" == "11.8" (
        echo Installing PyTorch and ONNX Runtime with CUDA 11.8 support...
        uv pip install --python "%VENV_PYTHON%" torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
        uv pip install --python "%VENV_PYTHON%" onnxruntime-gpu==1.16.3
    ) else if "%CUDA_VERSION:~0,4%" == "11.7" (
        echo Installing PyTorch with CUDA 11.7 support...
        uv pip install --python "%VENV_PYTHON%" torch==2.0.1+cu117 torchvision==0.15.2+cu117 --index-url https://download.pytorch.org/whl/cu117
        uv pip install --python "%VENV_PYTHON%" onnxruntime-gpu==1.16.1
    ) else (
        echo CUDA version not explicitly supported, defaulting to CUDA 11.8...
        uv pip install --python "%VENV_PYTHON%" torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
    )
) else (
    echo No NVIDIA GPU detected, installing CPU-only versions...
    uv pip install --python "%VENV_PYTHON%" torch==2.0.1+cpu torchvision==0.15.2+cpu --index-url https://download.pytorch.org/whl/cpu
)

:: Install other requirements from requirements.txt
echo Installing other requirements from requirements.txt...
uv pip install --python "%VENV_PYTHON%" -r "%SCRIPT_DIR%requirements.txt"

:: Create launch script if it doesn't exist
if not exist "%SCRIPT_DIR%\start.bat" (
    echo Creating launch script...
    (
    echo @echo off
    echo cd "%SCRIPT_DIR%"
    echo "%VENV_PYTHON%" run.py
    echo pause
    ) > "%SCRIPT_DIR%\start.bat"
)

echo Setup complete!
echo Run start.bat to launch the application.
pause