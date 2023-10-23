:INIT

@echo off

echo:
echo *** DELETING CONDA/PYTHON VIRTUAL ENVIRONMENT FOR PYRL ***
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
   echo   "venv_remove test" for deleting pyrl_test
   echo   "venv_remove dev" for deleting pyrl_dev
   echo Setting to "test".
   set VENV_PYRL_NAME=pyrl_test
   echo:
)

:PATH

set OLD_PATH=%PATH%
set CONDA_ADD_PATH=%CONDA_PATH%;%CONDA_PATH%\Library\mingw-w64\bin;%CONDA_PATH%\Library\usr\bin;%CONDA_PATH%\Library\bin;%CONDA_PATH%\Scripts;%CONDA_PATH%\bin;%CONDA_PATH%\condabin
set PATH=%CONDA_ADD_PATH%;%PATH%
set CONDA_ADD_PATH=

:MAIN

echo:
echo - Deleting environment "%VENV_PYRL_NAME%"
echo:

REM %CONDA_PATH%\Scripts\conda remove -n %VENV_PYRL_NAME% --all
REM cmd /C %CONDA_PATH%\condabin\conda remove %VENV_PYRL_NAME%
REM cmd /C conda remove -n %VENV_PYRL_NAME% --all
cmd /C conda remove -n %VENV_PYRL_NAME% --all

echo:
del /s /q %CONDA_PATH%\envs\%VENV_PYRL_NAME%\*.*
rmdir /s /q %CONDA_PATH%\envs\%VENV_PYRL_NAME%

REM echo:
REM conda info --envs

echo:
echo Operation terminated.
pause
echo: