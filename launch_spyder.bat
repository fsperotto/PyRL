@echo off

set CONDA_PATH=D:\fperotto\miniconda3
set VENV_PATH=%CONDA_PATH%\envs\pyrl_dev
 
start %CONDA_PATH%\pythonw.exe %CONDA_PATH%\cwp.py %VENV_PATH% %VENV_PATH%\pythonw.exe %VENV_PATH%\Scripts\spyder-script.py
REM start spyder

set VENV_PATH=