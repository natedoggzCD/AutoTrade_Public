@echo off
setlocal enabledelayedexpansion
REM =====================================================
REM AUTONOMOUS AGENT MAIN - GUARDED SUPERVISED LAUNCHER
REM =====================================================
REM New hardened entrypoint for live 24/7 supervised runtime.
REM Leaves start_autonomous_agent.bat option 12 untouched.
REM =====================================================

cd /d "%~dp0"

REM H6 (handoff 2026-05-22) / H2 (2026-05-21): refuse to launch elevated.
REM Running the agent at admin/elevated integrity prevents the non-elevated
REM day-manager from killing the process tree to apply hot-fixes. The agent
REM only needs Alpaca HTTPS, Ollama (user-context), and user-owned file I/O —
REM admin is never required. If this check fires, relaunch from a normal
REM (non-elevated) shell or fix the shortcut/scheduled task that elevates.
net session >nul 2>&1
if %errorlevel% equ 0 (
    echo [ERROR] Detected elevated/Administrator shell.
    echo         The autonomous agent must run at user integrity so the
    echo         day-manager can restart it without admin rights.
    echo         Close this window and relaunch from a normal shell.
    pause
    exit /b 87
)

set "FRESH_FLAG="
if /I "%~1"=="--fresh" set "FRESH_FLAG=--fresh"
if /I "%~1"=="/fresh" set "FRESH_FLAG=--fresh"

echo ======================================================
echo  AUTONOMOUS AGENT MAIN - GUARDED SUPERVISED LAUNCH
echo ======================================================
echo  Starting supervised 24/7 time-aware mode...
echo  Legacy option 12 remains untouched as fallback.
echo ======================================================
echo.

if exist ".env" (
    echo [INFO] Loading environment from .env file...
    for /f "usebackq tokens=1,* delims==" %%a in (".env") do (
        set "line=%%a"
        if not "!line:~0,1!"=="#" if not "!line!"=="" (
            set "%%a=%%b"
        )
    )
    echo [OK] Environment variables loaded
) else (
    echo [WARNING] .env file not found
)

set "CONDA_ENV_PRIMARY=stocks-gpu"
set "CONDA_ENV=%CONDA_ENV_PRIMARY%"
set "CONDA_BASE=%USERPROFILE%\miniconda3"
set "CONDA_ACTIVATE=%CONDA_BASE%\Scripts\activate.bat"

echo [INFO] Activating conda environment: %CONDA_ENV_PRIMARY%
if exist "%CONDA_ACTIVATE%" (
    call "%CONDA_ACTIVATE%" %CONDA_ENV_PRIMARY% 2>nul
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to activate conda env "%CONDA_ENV_PRIMARY%".
        pause
        exit /b 1
    )
    echo [OK] Conda environment activated: !CONDA_ENV!
) else (
    echo [ERROR] Conda not found at: %CONDA_ACTIVATE%
    pause
    exit /b 1
)

echo [CHECK] Verifying SearXNG...
if not defined SEARXNG_URL set "SEARXNG_URL=http://localhost:8080"
curl -s "%SEARXNG_URL%/healthz" >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] SearXNG not running. Web research may be limited.
) else (
    echo [OK] SearXNG is running
)

echo [CHECK] Verifying Ollama...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Ollama not running. AI analysis may be limited.
) else (
    echo [OK] Ollama is running
)

echo [CHECK] Verifying Alpaca credentials...
if defined ALPACA_API_KEY (
    echo [OK] Alpaca API key found
) else (
    echo [WARNING] ALPACA_API_KEY not set. Position tracking may be limited.
)

REM Critical runtime repair is API-only. Local Ollama/qwen repair is disabled.
if not defined AUTOTRADE_CODE_REPAIR_API_ONLY set "AUTOTRADE_CODE_REPAIR_API_ONLY=1"
if not defined AUTOTRADE_ENABLE_QWEN3_REPAIR set "AUTOTRADE_ENABLE_QWEN3_REPAIR=0"
if not defined AUTOTRADE_ENABLE_OPENAI_REPAIR set "AUTOTRADE_ENABLE_OPENAI_REPAIR=1"
if not defined AUTOTRADE_ENABLE_CODEX_REPAIR set "AUTOTRADE_ENABLE_CODEX_REPAIR=0"
if not defined SELF_HEAL_ENABLE_LIVE set "SELF_HEAL_ENABLE_LIVE=0"

echo.
echo [SUPERVISOR] Launching guarded continuous mode...
if not "%FRESH_FLAG%"=="" (
    echo [SUPERVISOR] Fresh start requested
)

python -m autotrade.core.workflow_claw %FRESH_FLAG%
set "EXIT_CODE=%ERRORLEVEL%"

echo.
echo [RESULT] Guarded supervised launcher exited with code %EXIT_CODE%
pause
exit /b %EXIT_CODE%
