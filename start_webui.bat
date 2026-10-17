@echo off
rem Start the UniForge3D web UI with the 8 GB defaults: fp16 denoisers, each stage moved onto
rem the card only for its own steps. The weights load a stage at a time, so the first job
rem starts within seconds and the rest of the stack arrives while the GPU works. Every
rem stage's weights stay in RAM once loaded, which is what a 64 GB machine can hold.
rem
rem Extra flags are passed through, e.g.:  start_webui.bat --port 8000 --open
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [start] no .venv here. Run scripts\install_windows.ps1 first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" webui.py %*
set "status=%ERRORLEVEL%"
rem 0xC000013A is Ctrl+C, which is how the window is normally closed.
if not "%status%"=="0" if not "%status%"=="-1073741510" (
    echo [start] webui.py exited with %status%
    pause
)
exit /b %status%
