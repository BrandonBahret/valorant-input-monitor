@echo off
setlocal

REM ===============================
REM CONFIG
REM ===============================
set APP_NAME=InputMonitor
set ENTRY_POINT=input_monitor.py
set PNG_ICON=assets/favicon-512x512.png
set ICO_FILE=assets/favicon.ico
set OUTPUT_DIR=debug-build

REM Extract base name from entry point (input_monitor.py -> input_monitor)
for %%F in (%ENTRY_POINT%) do set DIST_FOLDER=%%~nF.dist

echo Checking source files...
if not exist "%PNG_ICON%" (
  echo ERROR: PNG source not found: %PNG_ICON%
  pause
  exit /b 1
)
if not exist "%ENTRY_POINT%" (
  echo ERROR: Entry point not found: %ENTRY_POINT%
  pause
  exit /b 1
)

REM ===============================
REM CLEAN OLD BUILDS
REM ===============================
if exist %OUTPUT_DIR% rmdir /s /q %OUTPUT_DIR%
if exist %DIST_FOLDER% rmdir /s /q %DIST_FOLDER%
if exist %APP_NAME%.exe del %APP_NAME%.exe

REM ===============================
REM CONDITIONAL ICON GENERATION
REM ===============================
if not exist "%ICO_FILE%" (
  echo Generating icon file...
  python generate_icon.py
  if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Icon generation script failed with code %ERRORLEVEL%
    pause
    exit /b 1
  )
  if not exist "%ICO_FILE%" (
    echo ERROR: Icon generation failed - file not created!
    pause
    exit /b 1
  )
  echo Icon generated successfully: %ICO_FILE%
)

REM ===============================
REM BUILD WITH NUITKA
REM ===============================
echo.
echo Building with Nuitka (DEBUG MODE)...
python -m nuitka ^
  --mode=standalone ^
  --windows-console-mode=disable ^
  --windows-icon-from-ico=%ICO_FILE% ^
  --include-data-files=%PNG_ICON%=assets/favicon-512x512.png ^
  --output-filename=%APP_NAME%.exe ^
  --assume-yes-for-downloads ^
  --lto=no ^
  --show-progress ^
  --show-memory ^
  %ENTRY_POINT%

if %ERRORLEVEL% NEQ 0 (
  echo ERROR: Nuitka build failed with code %ERRORLEVEL%
  pause
  exit /b 1
)

REM ===============================
REM MOVE TO DEBUG-BUILD DIRECTORY
REM ===============================
if exist %DIST_FOLDER% (
  if not exist %OUTPUT_DIR% mkdir %OUTPUT_DIR%
  move %DIST_FOLDER% %OUTPUT_DIR%\
  echo.
  echo ===============================
  echo DEBUG Build finished!
  echo Output: %OUTPUT_DIR%\%DIST_FOLDER%\
  echo Run: %OUTPUT_DIR%\%DIST_FOLDER%\%APP_NAME%.exe
  echo ===============================
) else (
  echo.
  echo ===============================
  echo Build failed! Build directory not found.
  echo Expected: %DIST_FOLDER%
  echo ===============================
)

pause