@echo off
rem Delete the half-precision copies made by convert_weights_fp16.bat and point
rem weights\pipeline.local.json back at the fp32 originals. Nothing else is touched, and the
rem copies can be rebuilt at any time by running convert_weights_fp16.bat again.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [fp16] no .venv here. Run scripts\install_windows.ps1 first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" scripts\convert_weights_fp16.py --revert
set "status=%ERRORLEVEL%"
if not "%status%"=="0" (
    echo.
    echo [fp16] revert failed with %status%
    pause
    exit /b %status%
)

echo.
echo [fp16] reverted. infer.py, webui.py and both start_webui bats use the fp32 originals again.
pause
