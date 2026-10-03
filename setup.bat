@echo off
setlocal
cd /d "%~dp0"

echo === RoadDelta setup ===

REM ---------- folders ----------
if not exist data\raw mkdir data\raw

REM ---------- python env ----------
where python >nul 2>nul
if errorlevel 1 (
  echo Python not found. Install Python 3.10 or newer, then re-run.
  exit /b 1
)

if not exist .venv python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

REM PyTorch is not installed here because the right build depends on your GPU.
REM Get the install command from https://pytorch.org/get-started/locally/

REM ---------- MUNO21 data (about 14 GB) ----------
if not exist data\raw\mapupdate (
  if not exist data\raw\muno21.zip (
    call :download
    if errorlevel 1 exit /b 1
  )
  echo Extracting MUNO21. This takes a few minutes.
  tar -xf data\raw\muno21.zip -C data\raw
  if errorlevel 1 (
    echo Extraction failed. Check free disk space, then re-run.
    exit /b 1
  )
)

REM ---------- sanity check ----------
if exist data\raw\mapupdate python scripts\inspect_data.py --root data\raw\mapupdate

echo.
echo === Setup finished ===
exit /b 0

REM ---------- resumable download ----------
REM Downloads to muno21.zip.part and renames to muno21.zip only when complete.
REM Safe to stop and re-run: it continues from where it left off.
:download
echo Downloading MUNO21, about 14 GB. If the connection drops this retries and resumes.
set /a TRIES=0
:retry
set /a TRIES+=1
curl -L -C - --retry 10 --retry-delay 5 --retry-all-errors -o data\raw\muno21.zip.part https://favyen.com/files/muno21.zip
if errorlevel 1 (
  if %TRIES% geq 50 (
    echo Gave up after 50 attempts. Re-run later, or download the file in a browser.
    exit /b 1
  )
  echo Connection dropped. Retrying in 10 seconds, attempt %TRIES% of 50.
  timeout /t 10 /nobreak >nul
  goto retry
)
move /y data\raw\muno21.zip.part data\raw\muno21.zip >nul
exit /b 0