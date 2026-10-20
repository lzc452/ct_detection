@echo off
cd /d "%~dp0"
title CT缺陷检测系统 - Web端

echo ============================================
echo   CT缺陷检测系统 - Web端
echo ============================================
echo.

if not exist "runtime\python.exe" (
    echo [错误] 未找到便携运行环境 runtime\python.exe
    echo 请确认整个文件夹是完整拷贝的，不要只拷贝 bat 文件。
    pause
    exit /b 1
)

if "%PROCESSOR_ARCHITECTURE%"=="" set PROCESSOR_ARCHITECTURE=AMD64

echo 正在启动，请稍等 10-20 秒（模型预热中）...
echo 浏览器会自动打开 http://127.0.0.1:5000/
echo 关闭本窗口即可停止服务。
echo.

runtime\python.exe web\app.py

if errorlevel 1 (
    echo.
    echo [错误] 程序异常退出！请把上方错误信息截图发给技术支持。
)
echo.
pause
