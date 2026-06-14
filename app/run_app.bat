@echo off
title Video Analytics Tool Control Panel
echo ===================================================
echo Starting Video Analytics Tool - Presentation Mode
echo ===================================================

:: CONDA HUNTER: Find the Conda activation script automatically
set CONDA_ACTIVATE="%USERPROFILE%\miniconda3\Scripts\activate.bat"
if not exist %CONDA_ACTIVATE% set CONDA_ACTIVATE="%USERPROFILE%\anaconda3\Scripts\activate.bat"
if not exist %CONDA_ACTIVATE% set CONDA_ACTIVATE="%ALLUSERSPROFILE%\miniconda3\Scripts\activate.bat"
if not exist %CONDA_ACTIVATE% set CONDA_ACTIVATE="%ALLUSERSPROFILE%\anaconda3\Scripts\activate.bat"

echo [1/3] Launching FastAPI Backend on Port 8080...
:: Start the backend using the exact path to Conda
start "FastAPI Backend Server" cmd /k "call %CONDA_ACTIVATE% video_tool_env && uvicorn main:app --port 8080 --reload"

echo [2/3] Waiting 5 seconds for CLIP model and server initialization...
timeout /t 5 /nobreak > NUL

echo [3/3] Launching Streamlit Frontend UI...
:: Start the frontend using the exact path to Conda
call %CONDA_ACTIVATE% video_tool_env
streamlit run ui.py