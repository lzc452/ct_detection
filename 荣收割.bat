@echo off
chcp 936 >/dev/null
cd /d "%~dp0"
title CT收割-荣(新型号) V4
if "%PROCESSOR_ARCHITECTURE%"=="" set PROCESSOR_ARCHITECTURE=AMD64
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ============================================
echo   荣(新型号) 全量收割 - 约31万张正面图
echo   断点续跑：中断后再双击本文件接着跑
echo   跑完后自动进入精选导出（每组Top12高置信）
echo ============================================
echo.

runtime\python.exe scripts\harvest_rong.py
runtime\python.exe scripts\harvest_rong.py --export

echo.
echo ============================================
echo   完成！出口：收割结果_荣1123_V4 文件夹
echo   X-AnyLabeling打开核对：误报删框，真缺陷精修
echo ============================================
pause
