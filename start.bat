@echo off
set "ROOT=%~dp0"
if not exist "%ROOT%.venv\Scripts\python.exe" (
    echo No .venv found in %ROOT% - running install.bat first...
    call "%ROOT%install.bat"
)
cd /d "%ROOT%"
"%ROOT%.venv\Scripts\python.exe" run.py
pause
