@echo off
cd /d "%~dp0"
title CT缺陷检测系统 - 模型训练

echo ============================================
echo   CT缺陷检测系统 - 模型训练
echo ============================================
echo.

if not exist "runtime\python.exe" (
    echo [错误] 未找到便携运行环境 runtime\python.exe
    echo 请确认整个文件夹是完整拷贝的，不要只拷贝 bat 文件。
    pause
    exit /b 1
)

if "%PROCESSOR_ARCHITECTURE%"=="" set PROCESSOR_ARCHITECTURE=AMD64

echo 接下来会列出 datasets 里的数据集，输入编号选择训练集。
echo 有显卡自动用显卡训练，没有则用 CPU（会慢很多）。
echo 中途可按 Ctrl+C 停止，进度保存在 runs 文件夹。
echo.
pause

runtime\python.exe scripts\train.py

echo.
echo 训练结束！确认效果满意后，把 best.pt 手动复制到 models\启用\ 即可在网页选用。
pause
