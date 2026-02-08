@echo off
setlocal

REM ===============================
REM CONFIG
REM ===============================
set APP_NAME=InputMonitor
set ENTRY_POINT=input_monitor.py
set PNG_ICON=assets/favicon-512x512.png
set ICO_FILE=assets/favicon.ico

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
REM CLEAN OLD OUTPUTS (keep build cache!)
REM ===============================
if exist dist rmdir /s /q dist
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
REM BUILD WITH NUITKA (PRODUCTION)
REM ===============================
echo.
echo Building with Nuitka (PRODUCTION - onefile mode)...

python -m nuitka ^
  --mode=onefile ^
  --windows-console-mode=disable ^
  --windows-icon-from-ico=%ICO_FILE% ^
  --include-data-files=%PNG_ICON%=assets/favicon-512x512.png ^
  --output-filename=%APP_NAME%.exe ^
  --assume-yes-for-downloads ^
  %ENTRY_POINT%

if %ERRORLEVEL% NEQ 0 (
  echo ERROR: Nuitka build failed with code %ERRORLEVEL%
  pause
  exit /b 1
)

REM ===============================
REM MOVE TO DIST DIRECTORY
REM ===============================
if exist %APP_NAME%.exe (
  if not exist dist mkdir dist
  move %APP_NAME%.exe dist\
  echo.
  echo ===============================
  echo PRODUCTION Build finished!
  echo Output: dist\%APP_NAME%.exe
  echo Icon: %ICO_FILE% ^(for executable^)
  echo Runtime Icon: favicon-512x512.png ^(embedded^)
  echo ===============================
) else (
  echo.
  echo ===============================
  echo Build failed! Executable not found.
  echo Expected: %APP_NAME%.exe
  echo ===============================
)

pause