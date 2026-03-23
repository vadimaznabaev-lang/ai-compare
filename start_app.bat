@echo off
cd /d "%~dp0"

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python not found. Please install Python or add it to PATH.
    pause
    exit /b 1
)

start "" http://localhost:8501
python -m streamlit run app.py --server.headless true
