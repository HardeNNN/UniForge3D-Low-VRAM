@echo off
rem Start the UniForge3D web UI in low-RAM mode: the same 8 GB VRAM defaults, but only the
rem stage being run is held in RAM. Peak host memory becomes one group of weights instead of
rem the whole 25 GiB stack, at the cost of reading the weights again for every job (~70 s
rem per case on the test machine). For machines with less than 64 GB of RAM.
rem
rem Extra flags are passed through, e.g.:  start_webui_lowram.bat --port 8000 --open
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [start] no .venv here. Run scripts\install_windows.ps1 first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" webui.py --low-ram %*
set "status=%ERRORLEVEL%"
rem 0xC000013A is Ctrl+C, which is how the window is normally closed.
if not "%status%"=="0" if not "%status%"=="-1073741510" (
    echo [start] webui.py exited with %status%
    pause
)
exit /b %status%
