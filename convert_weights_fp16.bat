@echo off
rem Write half-precision copies of the fp32 checkpoints the pipeline halves anyway, verify
rem them, and switch weights\pipeline.local.json over to them. Saves about 10 GB of disk and
rem makes every run read and hold less of the checkpoint bytes. Takes a minute or two.
rem
rem The originals are never modified. revert_weights_fp16.bat undoes this.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [fp16] no .venv here. Run scripts\install_windows.ps1 first.
    pause
    exit /b 1
)
if not exist "weights\pipeline.local.json" (
    echo [fp16] no weights\pipeline.local.json yet.
    echo [fp16] run:  .venv\Scripts\python.exe scripts\make_local_config.py
    pause
    exit /b 1
)

".venv\Scripts\python.exe" scripts\convert_weights_fp16.py %*
set "status=%ERRORLEVEL%"
if not "%status%"=="0" (
    echo.
    echo [fp16] conversion failed with %status%
    pause
    exit /b %status%
)

echo.
echo [fp16] done. weights\pipeline.local.json now uses the copies, so infer.py, webui.py and
echo [fp16] both start_webui bats pick them up without extra flags.
echo [fp16] the thorough check (about five minutes):
echo [fp16]   .venv\Scripts\python.exe scripts\check_fp16_weights.py
pause
