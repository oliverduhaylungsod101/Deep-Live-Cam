@echo off
setlocal EnableExtensions
cd /d "%~dp0.."

set "PY=venv\Scripts\python.exe"
set "PI=venv\Scripts\pyinstaller.exe"
if not exist "%PY%" (
  echo Missing venv. Create it and install requirements first.
  exit /b 1
)

echo Installing/updating PyInstaller...
"%PY%" -m pip install "pyinstaller>=6.0" -q
if errorlevel 1 exit /b 1

echo Cleaning previous build...
if exist build rmdir /s /q build
if exist dist\Deep-Live-Cam rmdir /s /q dist\Deep-Live-Cam
if exist deploy\Deep-Live-Cam rmdir /s /q deploy\Deep-Live-Cam

echo Building run.exe (onedir)...
"%PI%" --noconfirm --clean packaging\run.spec
if errorlevel 1 (
  echo PyInstaller build failed.
  exit /b 1
)

echo Assembling deploy folder...
mkdir deploy 2>nul
xcopy /E /I /Y "dist\Deep-Live-Cam" "deploy\Deep-Live-Cam\" >nul

rem Models are large — copy if present (required at runtime).
if exist models (
  echo Copying models...
  xcopy /E /I /Y "models" "deploy\Deep-Live-Cam\models\" >nul
  rem Drop HF hub cache duplicate and obsolete Vosk small model from deploy.
  if exist "deploy\Deep-Live-Cam\models\models--Systran--faster-whisper-small.en" (
    rmdir /s /q "deploy\Deep-Live-Cam\models\models--Systran--faster-whisper-small.en"
  )
  if exist "deploy\Deep-Live-Cam\models\vosk-model-small-en-us-0.15" (
    rmdir /s /q "deploy\Deep-Live-Cam\models\vosk-model-small-en-us-0.15"
  )
)

if not exist "deploy\Deep-Live-Cam\models\faster-whisper-small.en\model.bin" (
  echo WARNING: Whisper caption model missing ^(models\faster-whisper-small.en^).
)

rem InsightFace buffalo_l (face detection) — required, no online download in frozen builds.
set "BUFFALO=%USERPROFILE%\.insightface\models\buffalo_l"
if exist "%BUFFALO%\det_10g.onnx" (
  echo Copying InsightFace buffalo_l...
  mkdir "deploy\Deep-Live-Cam\models\buffalo_l" 2>nul
  xcopy /E /I /Y "%BUFFALO%" "deploy\Deep-Live-Cam\models\buffalo_l\" >nul
) else if exist "models\buffalo_l\det_10g.onnx" (
  echo Using bundled models\buffalo_l...
) else (
  echo WARNING: buffalo_l not found. Face detection will fail offline.
)

rem Ensure ffmpeg is present.
if exist ffmpeg\bin (
  xcopy /E /I /Y "ffmpeg\bin" "deploy\Deep-Live-Cam\ffmpeg\bin\" >nul
)

copy /Y "packaging\DEPLOY_README.txt" "deploy\Deep-Live-Cam\README_DEPLOY.txt" >nul

echo.
echo Deploy ready: deploy\Deep-Live-Cam\run.exe
echo Double-click run.exe (defaults to --execution-provider cuda)
endlocal
