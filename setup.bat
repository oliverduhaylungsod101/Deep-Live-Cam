@echo off
setlocal EnableExtensions
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
  echo Python was not found on PATH.
  echo Install Python 3.11 or newer, then run setup.bat again.
  exit /b 1
)

python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
if errorlevel 1 (
  echo Python 3.11 or newer is required.
  python --version
  exit /b 1
)

if exist "venv\Scripts\python.exe" goto install_packages

echo Creating virtual environment...
python -m venv venv
if errorlevel 1 exit /b 1

:install_packages
echo Installing Python packages...
venv\Scripts\python.exe -m pip install -U pip
if errorlevel 1 exit /b 1
venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

if exist "ffmpeg\bin\ffmpeg.exe" goto download_models

echo Downloading FFmpeg...
if not exist "ffmpeg\bin" mkdir "ffmpeg\bin"
set "FFMPEG_ZIP=%TEMP%\ffmpeg-release-essentials.zip"
set "FFMPEG_EXTRACT=%TEMP%\ffmpeg-essentials-extract"
if exist "%FFMPEG_EXTRACT%" rmdir /s /q "%FFMPEG_EXTRACT%"
mkdir "%FFMPEG_EXTRACT%"
curl -L --fail -o "%FFMPEG_ZIP%" https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip
if errorlevel 1 (
  echo FFmpeg download failed.
  exit /b 1
)
tar -xf "%FFMPEG_ZIP%" -C "%FFMPEG_EXTRACT%"
if errorlevel 1 (
  echo FFmpeg extract failed.
  exit /b 1
)
set "FFMPEG_COPIED="
for /d %%D in ("%FFMPEG_EXTRACT%\ffmpeg-*") do (
  copy /Y "%%D\bin\ffmpeg.exe" "ffmpeg\bin\" >nul
  copy /Y "%%D\bin\ffprobe.exe" "ffmpeg\bin\" >nul
  if exist "%%D\bin\ffplay.exe" copy /Y "%%D\bin\ffplay.exe" "ffmpeg\bin\" >nul
  set "FFMPEG_COPIED=1"
)
if not exist "ffmpeg\bin\ffmpeg.exe" (
  echo FFmpeg zip did not contain ffmpeg.exe.
  exit /b 1
)

:download_models
echo Downloading face-swap models...
venv\Scripts\python.exe -c "from modules.model_registry import ensure_app_models; ensure_app_models(print)"
if errorlevel 1 exit /b 1

echo.
echo Setup finished.
echo Run the app with:  run-cuda.bat
echo An NVIDIA driver is required for CUDA. InsightFace and Whisper models download on first use.
exit /b 0
