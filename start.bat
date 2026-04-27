 @echo off
setlocal

cd /d "%~dp0"

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python virtual environment not found at .venv\Scripts\python.exe
    echo Please create and install dependencies first.
    echo Example:
    echo   python -m venv .venv
    echo   .venv\Scripts\activate
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)

echo Starting Meta-Guard backend and frontend...

start "Meta-Guard Backend" cmd /k ""%PYTHON_EXE%" -m uvicorn backend.app.main:app --reload --port 8080 --env-file .env"
start "Meta-Guard Frontend" cmd /k ""%PYTHON_EXE%" -m streamlit run frontend/streamlit_app.py"

echo Done. Two windows were opened:
echo   1. FastAPI backend on http://127.0.0.1:8080
echo   2. Streamlit frontend

endlocal