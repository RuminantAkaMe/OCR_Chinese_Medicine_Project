@echo off
cd /d %~dp0
start "" "http://localhost:5173"
npm run dev

REM Nur offen halten, wenn NICHT über start_all.bat gestartet:
if "%1" neq "silent" pause