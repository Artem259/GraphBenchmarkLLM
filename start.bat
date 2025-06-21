@echo off
set NUM_GRAPHS=2
set SCRIPT_PATH=main.py

for /l %%i in (1,1,3) do (
    echo Running iteration %%i...
    python %SCRIPT_PATH% %NUM_GRAPHS%
    echo.
)

echo Benchmarking complete.
pause
