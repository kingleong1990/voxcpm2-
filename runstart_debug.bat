@echo off
chcp 65001 >nul
echo ==========================================
echo VoxCPM2 - Debug Startup Script
echo ==========================================
echo.

REM Set environment variables
set "PYTHON_PATH=%cd%\python\"
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONEXECUTABLE=%PYTHON_PATH%python.exe"
set "PYTHONWEXECUTABLE=%PYTHON_PATH%pythonw.exe"
set "PYTHON_EXECUTABLE=%PYTHON_PATH%python.exe"
set "PYTHONW_EXECUTABLE=%PYTHON_PATH%pythonw.exe"
set "PYTHON_BIN_PATH=%PYTHON_EXECUTABLE%"
set "PYTHON_LIB_PATH=%PYTHON_PATH%Lib\site-packages"
set "FFMPEG_PATH=%cd%\python\Scripts"
set "PATH=%PYTHON_PATH%;%PYTHON_PATH%Scripts;%FFMPEG_PATH%;%PATH%"
set "HF_HOME=%cd%\hf_download"
set "MODEL_PATH=%cd%\models"
set "HF_ENDPOINT=https://hf-mirror.com"

echo [DEBUG] Current directory: %cd%
echo [DEBUG] PYTHON_PATH: %PYTHON_PATH%
echo [DEBUG] PYTHON_EXECUTABLE: %PYTHON_EXECUTABLE%
echo [DEBUG] PYTHON_LIB_PATH: %PYTHON_LIB_PATH%
echo.

REM ==========================================
REM Fix the editable install path in the .pth file
REM ==========================================
echo [0/4] Fixing module paths...
set "CURRENT_SRC_PATH=%~dp0src"
set "CURRENT_SRC_PATH=%CURRENT_SRC_PATH:\=/%"

REM Find and update any voxcpm .pth files
for %%f in ("%PYTHON_LIB_PATH%\__editable__.voxcpm-*.pth") do (
    echo [DEBUG] Updating: %%f
    echo %CURRENT_SRC_PATH% > "%%f"
    echo [OK] Updated .pth file to: %CURRENT_SRC_PATH%
)
echo.

REM Check 1: Python environment
echo [1/4] Checking Python environment...
if not exist "%PYTHON_EXECUTABLE%" (
    echo [ERROR] Python not found at: %PYTHON_EXECUTABLE%
    pause
    exit /b 1
)
echo [OK] Python found: %PYTHON_EXECUTABLE%
echo.

REM Check Python version
echo [DEBUG] Python version:
"%PYTHON_EXECUTABLE%" --version
echo.

REM Check 2: VoxCPM module
echo [2/4] Checking VoxCPM module...
pushd "%~dp0"
"%PYTHON_EXECUTABLE%" -c "import voxcpm; print('VoxCPM module successfully loaded')"
if errorlevel 1 (
    echo [ERROR] VoxCPM module not found!
    echo [DEBUG] Trying to find src folder...
    if exist "src\voxcpm\__init__.py" (
        echo [DEBUG] src\voxcpm\__init__.py exists
        echo [DEBUG] Trying to add src to PYTHONPATH...
        set "PYTHONPATH=%~dp0src"
        "%PYTHON_EXECUTABLE%" -c "import sys; sys.path.insert(0, r'%~dp0src'); import voxcpm; print('VoxCPM module loaded via PYTHONPATH')"
        if errorlevel 1 (
            echo [ERROR] Still cannot import voxcpm!
            popd
            pause
            exit /b 1
        )
    ) else (
        echo [ERROR] src\voxcpm\__init__.py NOT found!
        popd
        pause
        exit /b 1
    )
)
popd
echo [OK] VoxCPM module found
echo.

REM Check 3: Model files
echo [3/4] Checking model files...
set "MODEL_FOUND=0"
if exist "%cd%\models\openbmb__VoxCPM2" set "MODEL_FOUND=1"
if exist "%cd%\VoxCPM2.0" set "MODEL_FOUND=1"

if %MODEL_FOUND%==0 (
    echo [WARNING] VoxCPM2 model not found in 'models' folder!
    echo.
    choice /C YN /M "Continue anyway"
    if errorlevel 2 exit /b 1
) else (
    echo [OK] Model found
)
echo.

echo [4/4] Starting VoxCPM2...
echo.
echo The service will be available at:
echo   - Local access: http://127.0.0.1:8808
echo.

REM Change to script directory
pushd "%~dp0"

REM Start the service in background
echo [DEBUG] Starting app.py...
start "VoxCPM2" /MIN "%PYTHON_EXECUTABLE%" app.py --host 0.0.0.0 --port 8808

REM Wait for the server to start
echo Waiting for VoxCPM2 to start...
echo This may take 10-30 seconds...
timeout /t 10 /nobreak >nul

REM Try to get the local IP address and display it
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4 Address"') do set ipaddr=%%a
set ipaddr=%ipaddr: =%

REM Wait a bit more for Gradio to fully start
timeout /t 5 /nobreak >nul

REM Open browser to local address
start http://127.0.0.1:8808

REM Display the IP address if found
if defined ipaddr (
    echo.
    echo ==========================================
    echo Access VoxCPM2 from other devices: http://%ipaddr%:8808
    echo ==========================================
)

echo.
echo VoxCPM2 is running!
echo If the browser shows an error, wait a few seconds and refresh.
echo To stop the service, close the 'VoxCPM2' window in taskbar.
echo ==========================================
pause
