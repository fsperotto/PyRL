:INIT

@echo off

echo:
echo *** ACTIVATING CONDA/PYTHON VIRTUAL ENVIRONMENT FOR PYRL ***
echo:

REM if not defined CONDA_PATH set CONDA_PATH=C:\Anaconda3
if not defined CONDA_PATH set CONDA_PATH=D:\fperotto\miniconda3
echo CONDA_PATH is %CONDA_PATH%
if not defined PYRL_PATH set PYRL_PATH=.
echo PYRL_PATH is %PYRL_PATH%
echo:

:PARAMS

if [%1]==[dev] (
   set VENV_PYRL_NAME=pyrl_dev
) else (
   if not [%1]==[] echo Invalid option "%1"
   echo You should use:
   echo   "venv_activate test" for pyrl_test
   echo   "venv_activate dev" for pyrl_dev
   echo Setting to "test".
   set VENV_PYRL_NAME=pyrl_test
   echo:
)

REM set ANACONDA_ADD_PATH=%CONDA_PATH%;%CONDA_PATH%\Library\mingw-w64\bin;%CONDA_PATH%\Library\usr\bin;%CONDA_PATH%\Library\bin;%CONDA_PATH%\Scripts;%CONDA_PATH%\bin;%CONDA_PATH%\condabin
REM set PATH=%PATH%;%ANACONDA_ADD_PATH%

:MAIN

echo - Activating environment "%VENV_PYRL_NAME%"
cmd /K %CONDA_PATH%\Scripts\activate.bat %VENV_PYRL_NAME%
REM cmd /K %CONDA_PATH%\Scripts\activate.bat %CONDA_PATH%\envs\%VENV_NAME%
REM cmd /K %CONDA_PATH%\Scripts\conda.bat activate %VENV_NAME%
REM cmd /C conda activate %VENV_NAME%
REM cmd conda activate %VENV_NAME%
REM conda activate %VENV_NAME%
REM call cmd /C conda activate %VENV_NAME%
REM cmd /K conda activate %VENV_PYRL_NAME%
