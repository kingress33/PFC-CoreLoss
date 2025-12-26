@echo off
ECHO Starting Pytorch environment and Streamlit App...
ECHO.

REM 1. Activate the 'pytorch' environment using the FULL PATH.
REM    If your path is not 'C:\Users\user\anaconda3', YOU MUST CHANGE THIS LINE!
ECHO [Step 1] Activating 'pytorch' environment...
call C:\Users\user\anaconda3\Scripts\activate.bat pytorch

REM    Check if activation failed.
if %errorlevel% neq 0 (
    ECHO ERROR: Failed to activate Conda environment.
    ECHO Please check the path in this .bat file:
    ECHO C:\Users\user\anaconda3\Scripts\activate.bat
    ECHO.
    goto end
)

REM 2. Change directory to this batch file's location (e.g., your D: drive).
REM    %~dp0 is a variable for the batch file's current directory.
ECHO [Step 2] Changing directory...
cd /d %~dp0
ECHO Current directory: %cd%

REM 3. Run the Streamlit App.
ECHO [Step 3] Starting Tesnorboard (streamlit run ECIE-Home.py)...
ECHO.
tensorboard --logdir="D:\ProgramFiles\Jupyter\ECIE\Machine_Learning\MMINN\results"

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] TensorBoard 啟動失敗！
    pause
)

:end
ECHO.
ECHO Script finished. Press any key to exit.
pause