@echo off
rem Double-click to start CaddieAI: the API and the chat page each open
rem in their own window. Close both windows (or press Ctrl+C in each) to stop.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Missing .venv - follow the Setup steps in README.md first.
    pause
    exit /b 1
)

start "CaddieAI - API (keep open)" cmd /k ".venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8001"
rem Give the API a few seconds to load before the chat page asks it for status.
timeout /t 6 /nobreak >nul
start "CaddieAI - chat page (keep open)" cmd /k ".venv\Scripts\python.exe -m streamlit run frontend/streamlit_app.py"

echo Started. The chat page opens in your browser at http://localhost:8501
timeout /t 5 >nul
