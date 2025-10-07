@echo off
chcp 65001 > nul
title HWM Bot Launcher
echo Starting HWM Bot...
cd /d %~dp0
python main.py
if %errorlevel% neq 0 (
    echo Launch error! Check Python and dependencies.
)
pause