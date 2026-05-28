@echo off
setlocal enabledelayedexpansion
REM =====================================================
REM AUTONOMOUS TRADING AGENT - COMMAND CENTER (2026)
REM =====================================================
REM Version: 2.3 (2026-04-04)
REM Primary Env: stocks-gpu
REM =====================================================

title AutoTrade Command Center

cd /d "%~dp0"

REM --- SETTINGS & PATHS ---
if not defined PYTHON_EXE set "PYTHON_EXE=python"
set "CONDA_BASE=%USERPROFILE%\miniconda3"
set "CONDA_ACTIVATE=%CONDA_BASE%\Scripts\activate.bat"
set "CONDA_ENV=stocks-gpu"

REM --- LOGO ---
echo ======================================================
echo  AUTONOMOUS TRADING AGENT - COMMAND CENTER
echo  [2026 HIGH-FIDELITY ARCHITECTURE]
echo ======================================================

REM --- LOAD ENVIRONMENT ---
if exist ".env" (
    echo [INFO] Loading environment from .env file...
    for /f "usebackq tokens=1,* delims==" %%a in (".env") do (
        set "line=%%a"
        if not "!line:~0,1!"=="#" if not "!line!"=="" (
            set "%%a=%%b"
        )
    )
    echo [OK] Environment variables loaded
)

REM --- CONDA ACTIVATION ---
if exist "%CONDA_ACTIVATE%" (
    call "%CONDA_ACTIVATE%" %CONDA_ENV% 2>nul
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to activate conda env "%CONDA_ENV%".
        pause
        exit /b 1
    )
    echo [OK] Conda environment activated: !CONDA_ENV!
)

REM --- PRE-FLIGHT HYGIENE ---
echo [HYGIENE] Running session hygiene check...
"%PYTHON_EXE%" tools/session_hygiene_check.py --strict

REM --- SERVICE HEALTH ---
echo.
echo [CHECK] Verifying SearXNG...
curl -fsS http://localhost:8080/healthz >nul 2>&1
if !errorlevel! neq 0 (
    echo [WARN] SearXNG not reachable. Attempting Docker start...
    docker start searxng >nul 2>&1
    timeout /t 3 >nul
)

echo [CHECK] Verifying Ollama...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Ollama not running!
) else (
    echo [OK] Ollama is active
)

REM --- MAIN MENU ---
:MENU
cls
echo ======================================================
echo  AUTONOMOUS TRADING AGENT - MAIN MENU
echo ======================================================
echo.
echo  [1]  Show Current Positions and Odd Lots
echo  [2]  Generate Tomorrow's Trade Plan
echo  [3]  Research a Specific Symbol
echo  [4]  Run Quick Overnight Analysis (15 top picks)
echo  [5]  Run Premarket Analysis
echo  [6]  Run Daily Review (How did today go?)
echo  [7]  Show Chart Index Status
echo  [8]  List Available Analysis Tools
echo  [9]  Show Orchestrator Status
echo.
echo  [10] Run FULL Overnight Research (Scan 4448 - Best 200) [SELF-HEALING]
echo  [11] Run Continuous Mode (LIVE TRADING - REAL MONEY!)
echo  [12] Run Time-Aware Mode (Auto-timing + Continuous Research)
echo.
echo  [13] Validate PM Workflow
echo  [14] Optimize Watchlist
echo  [15] IMPROVE STRATEGY (DuckDB-Powered)
echo  [16] Workflow Validation (Dry Run)
echo  [17] YouTube Sentiment Analysis (PM Enhancement)
echo  [18] Weekend YouTube Intelligence (Deep Dive + OpenAI Synthesis)
echo  [19] Weekend Mode (Monday Prep: Overnight Scan + Watchlist)
echo  [20] Strategy Lab + Signal Factory (Backtest-Driven)
echo.
echo  [21] Diagnose DayManager (Execution Integrity)
echo  [22] Verify Live Connection (Data Freshness)
echo  [23] Rebuild Research State (Repair Corrupted Files)
echo.
echo  [99] Project Chat - Development Assistant (RAG)
echo.
echo  [0]  Exit
echo.
echo ======================================================
set /p "CHOICE=Enter your choice (0-23, 99): "

if "%CHOICE%"=="0" goto :END
if "%CHOICE%"=="1" goto :POSITIONS
if "%CHOICE%"=="2" goto :PLAN
if "%CHOICE%"=="3" goto :RESEARCH
if "%CHOICE%"=="4" goto :OVERNIGHT
if "%CHOICE%"=="5" goto :PREMARKET
if "%CHOICE%"=="6" goto :PMWORKFLOW
if "%CHOICE%"=="7" goto :CHARTS
if "%CHOICE%"=="8" goto :TOOLS
if "%CHOICE%"=="9" goto :STATUS
if "%CHOICE%"=="10" goto :CONTINUOUS_DRY
if "%CHOICE%"=="11" goto :CONTINUOUS_LIVE
if "%CHOICE%"=="12" goto :SCHEDULED
if "%CHOICE%"=="13" goto :VALIDATE
if "%CHOICE%"=="14" goto :OPTIMIZE
if "%CHOICE%"=="15" goto :IMPROVE
if "%CHOICE%"=="16" goto :WORKFLOW_VALIDATE
if "%CHOICE%"=="17" goto :YOUTUBE
if "%CHOICE%"=="18" goto :YOUTUBE_WEEKEND
if "%CHOICE%"=="19" goto :WEEKEND_MODE
if "%CHOICE%"=="20" goto :STRATEGY_LAB
if "%CHOICE%"=="21" goto :DIAGNOSE
if "%CHOICE%"=="22" goto :VERIFY_LIVE
if "%CHOICE%"=="23" goto :REBUILD_STATE
if "%CHOICE%"=="99" goto :PROJECTCHAT

echo [ERROR] Invalid choice. Please try again.
timeout /t 2 >nul
goto :MENU

REM --- TASK HANDLERS ---

:POSITIONS
echo.
"%PYTHON_EXE%" -m autotrade.core.master_supervisor --task positions --execute
goto :CONTINUE

:PLAN
echo.
"%PYTHON_EXE%" -m autotrade.core.master_supervisor --task plan --execute
goto :CONTINUE

:RESEARCH
echo.
set /p "SYM=Enter symbol to research (e.g., AAPL): "
"%PYTHON_EXE%" -m autotrade.core.master_supervisor --task research %SYM% --execute
goto :CONTINUE

:OVERNIGHT
echo.
"%PYTHON_EXE%" -m autotrade.core.master_supervisor --task overnight --execute
goto :CONTINUE

:PREMARKET
echo.
"%PYTHON_EXE%" -m autotrade.core.master_supervisor --task premarket --execute
goto :CONTINUE

:PMWORKFLOW
echo.
"%PYTHON_EXE%" -m autotrade.core.master_supervisor --task pm-workflow --execute
goto :CONTINUE

:CHARTS
echo.
"%PYTHON_EXE%" -m autotrade.core.master_supervisor --task charts --execute
goto :CONTINUE

:TOOLS
echo.
"%PYTHON_EXE%" -m autotrade.core.master_supervisor --task tools --execute
goto :CONTINUE

:STATUS
echo.
"%PYTHON_EXE%" -m autotrade.core.master_supervisor --task status --execute
goto :CONTINUE

:CONTINUOUS_DRY
echo.
echo [SUPERVISOR] Starting FULL Overnight Research (Scan ALL 4448)...
"%PYTHON_EXE%" -m autotrade.core.master_supervisor --task continuous --fresh --execute
goto :CONTINUE

:CONTINUOUS_LIVE
echo.
echo [!] WARNING: LIVE TRADING MODE [!]
set /p "CONFIRM=Type YES to confirm real money execution: "
if /i "%CONFIRM%"=="YES" (
    "%PYTHON_EXE%" -m autotrade.core.master_supervisor --task continuous --execute -- --execute
)
goto :CONTINUE

:SCHEDULED
echo.
echo [WATCHDOG] Starting 24/7 Time-Aware Mode...
set /a RESTART_COUNT=0
:WATCHDOG_LOOP
set /a RESTART_COUNT+=1
echo [WATCHDOG] Launch #%RESTART_COUNT% at %TIME%
if %RESTART_COUNT% EQU 1 (
    "%PYTHON_EXE%" -m autotrade.core.master_supervisor --task continuous --fresh --execute -- --execute --no-deadline
) else (
    "%PYTHON_EXE%" -m autotrade.core.master_supervisor --task continuous --execute -- --execute --no-deadline
)
echo [WATCHDOG] Agent exited. Restarting in 15s...
timeout /t 15 /nobreak >nul
goto :WATCHDOG_LOOP

:VALIDATE
echo.
"%PYTHON_EXE%" -m autotrade.core.master_supervisor --task validate --execute
goto :CONTINUE

:OPTIMIZE
echo.
"%PYTHON_EXE%" -m autotrade.core.master_supervisor --task optimize --execute
goto :CONTINUE

:IMPROVE
echo.
"%PYTHON_EXE%" -m autotrade.core.master_supervisor --task improve --execute
goto :CONTINUE

:WORKFLOW_VALIDATE
echo.
"%PYTHON_EXE%" -m autotrade.core.autonomous_agent --workflow-validate
goto :CONTINUE

:YOUTUBE
echo.
"%PYTHON_EXE%" tools/youtube_sentiment.py --dir data/youtube
goto :CONTINUE

:YOUTUBE_WEEKEND
echo.
"%PYTHON_EXE%" tools/youtube_weekend_scanner.py --verbose
goto :CONTINUE

:WEEKEND_MODE
echo.
echo [SUPERVISOR] Starting FULL Weekend Mode sequence...
echo (YouTube Intelligence -^> Overnight Research -^> Monday Plan)
"%PYTHON_EXE%" -m autotrade.core.master_supervisor --task weekend --execute
goto :CONTINUE

:STRATEGY_LAB
echo.
echo [STRATEGY LAB] Running autonomous signal factory...
"%PYTHON_EXE%" tools/strategy_lab.py auto --lookback-days 730 --universe-size 0 --top-candidates 3 --walkforward-lookback-days 730
goto :CONTINUE

:DIAGNOSE
echo.
"%PYTHON_EXE%" tools/diagnose_daymanager.py
goto :CONTINUE

:VERIFY_LIVE
echo.
"%PYTHON_EXE%" tools/verify_live.py
goto :CONTINUE

:REBUILD_STATE
echo.
"%PYTHON_EXE%" tools/rebuild_historical_overnight_state.py
goto :CONTINUE

:PROJECTCHAT
echo.
"%PYTHON_EXE%" tools/project_chat_bootstrap.py --mode improve
goto :CONTINUE

:CONTINUE
echo.
echo ======================================================
set /p "AGAIN=Return to menu? (Y/N): "
if /i "%AGAIN%"=="Y" goto :MENU
if /i "%AGAIN%"=="YES" goto :MENU

:END
echo.
echo Goodbye!
timeout /t 1 >nul
exit /b 0
