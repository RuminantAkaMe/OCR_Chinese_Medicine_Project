@echo off
echo Starte Backend...
start "Backend" cmd /k "cd /d %~dp0backend && call .venv310\Scripts\activate && uvicorn app.main:app --reload"

echo Starte Frontend...
start "Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

timeout /t 2 > nul

start "" "http://localhost:5173"

pause
