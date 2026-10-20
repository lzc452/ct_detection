@echo off
chcp 936 >nul
cd /d "%~dp0"
title CT产线收割机 - V4筛损伤原图(阈值0.40)
if "%PROCESSOR_ARCHITECTURE%"=="" set PROCESSOR_ARCHITECTURE=AMD64
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ============================================
echo   CT产线收割机（V4模型，导出损伤^>=0.40原图）
echo ============================================
echo.
echo  源文件夹结构：大文件夹/电芯组/正面/图
echo  断点续跑：中断后再双击会从上次进度继续
echo  已人工核过的老图自动跳过，不会重复导出
echo.
set "SRC=%~1"
if "%SRC%"=="" set /p SRC=源文件夹路径:
if "%SRC%"=="" (echo 未输入路径 ^& pause ^& exit /b 1)

runtime\python.exe scripts\harvest.py "%SRC%" 0.4
echo.
echo ============================================
echo  出口：收割结果_xxx_V4 文件夹（极片损伤）
echo  人工核对后放入样本库即可参与下轮训练
echo  没跑完的话再双击本文件自动续跑
echo ============================================
pause
