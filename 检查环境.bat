@echo off
cd /d "%~dp0"
title CT缺陷检测系统 - 环境体检

echo ============================================
echo   CT缺陷检测系统 - 环境体检
echo ============================================
echo.

if not exist "runtime\python.exe" (
    echo [错误] 未找到便携运行环境 runtime\python.exe
    echo 请确认整个文件夹是完整拷贝的，不要只拷贝 bat 文件。
    pause
    exit /b 1
)

if "%PROCESSOR_ARCHITECTURE%"=="" set PROCESSOR_ARCHITECTURE=AMD64

runtime\python.exe scripts\检查环境.py

echo.
pause
