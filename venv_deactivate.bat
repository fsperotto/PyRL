:INIT

@echo off

REM set CONDA_PATH=C:\Anaconda3

REM set VENV_NAME=pyrl

REM set ANACONDA_ADD_PATH=%CONDA_PATH%;%CONDA_PATH%\Library\mingw-w64\bin;%CONDA_PATH%\Library\usr\bin;%CONDA_PATH%\Library\bin;%CONDA_PATH%\Scripts;%CONDA_PATH%\bin;%CONDA_PATH%\condabin
REM set PATH=%PATH%;%ANACONDA_ADD_PATH%

:MAIN

REM echo - Deactivating environment "%VENV_NAME%"
REM cmd /K %CONDA_PATH%\Scripts\deactivate.bat %CONDA_PATH%\envs\%VENV_NAME%
cmd /K conda deactivate