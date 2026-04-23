@echo off
chcp 65001 >nul
title 商品图片净化工具 - 后端服务

echo.
echo  ================================================
echo    商品图片净化工具 - 启动中...
echo  ================================================
echo.

cd /d "%~dp0backend"

:: 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  [错误] 未找到 Python，请先安装 Python 3.8+
    echo  下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

:: 创建虚拟环境（首次）
if not exist "venv" (
    echo  [1/3] 正在创建虚拟环境...
    python -m venv venv
)

:: 激活虚拟环境
call venv\Scripts\activate.bat

:: 安装依赖（首次或更新）
echo  [2/3] 检查并安装依赖（首次约需 3-10 分钟，请耐心等待）...
pip install -r requirements.txt -q --no-warn-script-location

:: 启动服务
echo  [3/3] 正在启动服务...
echo.
echo  ================================================
echo    服务地址: http://127.0.0.1:5000
echo    前端页面: 双击打开 frontend\index.html
echo    按 Ctrl+C 停止服务
echo  ================================================
echo.

python app.py

pause
