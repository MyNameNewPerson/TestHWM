@echo off
title HWM Bot Launcher
echo Запуск HWM Bot...
cd d %~dp0
python main.py
if %errorlevel% neq 0 (
    echo Ошибка запуска! Проверьте Python и зависимости.
)
pause