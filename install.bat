@echo off
setlocal enabledelayedexpansion

:: Set paths
set "SCRIPT_DIR=%~dp0"
set "PYTHON_DIR=%SCRIPT_DIR%python"
set "PYTHON_URL=https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip"
set "GET_PIP_URL=https://bootstrap.pypa.io/get-pip.py"

:: Create necessary directories
mkdir "%SCRIPT_DIR%\logs" 2>nul
mkdir "%SCRIPT_DIR%\settings" 2>nul

echo Setting up environment in: %SCRIPT_DIR%

:: Check if Python is already installed
if exist "%PYTHON_DIR%\python.exe" (
    echo Python already installed, skipping download...
) else (
    :: Create directories if they don't exist
    if not exist "%PYTHON_DIR%" mkdir "%PYTHON_DIR%"

    :: Download Python embedded
    echo Downloading Python 3.11.9...
    powershell -Command "(New-Object Net.WebClient).DownloadFile('%PYTHON_URL%', '%SCRIPT_DIR%\python.zip')"
    echo Extracting Python...
    powershell -Command "Expand-Archive -Path '%SCRIPT_DIR%\python.zip' -DestinationPath '%PYTHON_DIR%' -Force"
    del "%SCRIPT_DIR%\python.zip"

    :: Enable pip in embedded Python
    echo Enabling pip...
    echo import site >> "%PYTHON_DIR%\python311._pth"

    :: Download and install pip
    echo Installing pip...
    powershell -Command "(New-Object Net.WebClient).DownloadFile('%GET_PIP_URL%', '%SCRIPT_DIR%\get-pip.py')"
    "%PYTHON_DIR%\python.exe" "%SCRIPT_DIR%\get-pip.py" --no-warn-script-location
    del "%SCRIPT_DIR%\get-pip.py"
)

echo Installing required packages...

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
        "%PYTHON_DIR%\python.exe" -m pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
        "%PYTHON_DIR%\python.exe" -m pip install onnxruntime-gpu==1.16.3
    ) else if "%CUDA_VERSION:~0,4%" == "11.7" (
        echo Installing PyTorch with CUDA 11.7 support...
        "%PYTHON_DIR%\python.exe" -m pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --index-url https://download.pytorch.org/whl/cu117
        "%PYTHON_DIR%\python.exe" -m pip install onnxruntime-gpu==1.16.1
    ) else (
        echo CUDA version not explicitly supported, defaulting to CUDA 11.8...
        "%PYTHON_DIR%\python.exe" -m pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
    )
) else (
    echo No NVIDIA GPU detected, installing CPU-only versions...
    "%PYTHON_DIR%\python.exe" -m pip install torch==2.0.1+cpu torchvision==0.15.2+cpu --index-url https://download.pytorch.org/whl/cpu
)

:: Install other requirements from requirements.txt
echo Installing other requirements from requirements.txt...
"%PYTHON_DIR%\python.exe" -m pip install -r "%SCRIPT_DIR%\requirements.txt"

:: Create launch script if it doesn't exist
if not exist "%SCRIPT_DIR%\start.bat" (
    echo Creating launch script...
    (
    echo @echo off
    echo cd "%SCRIPT_DIR%"
    echo "%PYTHON_DIR%\python.exe" run.py
    echo pause
    ) > "%SCRIPT_DIR%\start.bat"
)

echo Setup complete!
echo Run start.bat to launch the application.
pause