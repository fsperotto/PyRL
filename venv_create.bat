:INIT

@echo off

echo:
echo *** CREATING CONDA/PYTHON VIRTUAL ENVIRONMENT FOR PYRL ***
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
   set VENV_PACKAGES=conda pip sphynx spyder notebook
) else (
   if not [%1]==[] echo Invalid option "%1"
   echo You should use:
   echo   "venv_create test [pyrl_extension_options]" for creating pyrl_test
   echo   "venv_create dev [pyrl_extension_options]" for creating pyrl_dev
   echo Setting to "test".
   set VENV_PYRL_NAME=pyrl_test
   set VENV_PACKAGES=pip
   echo:
)

:PATH

set OLD_PATH=%PATH%
set CONDA_ADD_PATH=%CONDA_PATH%;%CONDA_PATH%\Library\mingw-w64\bin;%CONDA_PATH%\Library\usr\bin;%CONDA_PATH%\Library\bin;%CONDA_PATH%\Scripts;%CONDA_PATH%\bin;%CONDA_PATH%\condabin
set PATH=%CONDA_ADD_PATH%;%PATH%
set CONDA_ADD_PATH=

:MAIN

REM set VENV_ADD_PATH=%CONDA_PATH%\envs\%VENV_NAME%;%CONDA_PATH%\envs\%VENV_NAME%\Library\mingw-w64\bin;%CONDA_PATH%\envs\%VENV_NAME%\Library\usr\bin;%CONDA_PATH%\envs\%VENV_NAME%\Library\bin;%CONDA_PATH%\envs\%VENV_NAME%\Scripts;%CONDA_PATH%\envs\%VENV_NAME%\bin;%CONDA_PATH%\condabin
REM set PYTHON_VERSION=3.8

echo:
echo - Creating new environment "%VENV_PYRL_NAME%"
echo:

REM %CONDA_PATH%\Scripts\conda create --name %VENV_NAME% python=%PYTHON_VERSION% pip
REM %CONDA_PATH%\condabin\conda create --name %VENV_NAME% python=%PYTHON_VERSION% pip
REM %CONDA_PATH%\Scripts\conda create --name %VENV_NAME% python=%PYTHON_VERSION% anaconda
REM call conda create --name %VENV_NAME% python=%PYTHON_VERSION% anaconda
REM conda create --name %VENV_NAME% python=%PYTHON_VERSION% pip
REM conda create --prefix "%PYRL_PATH%\venvs\%VENV_PYRL_NAME%" pip
cmd /C conda create --name %VENV_PYRL_NAME% %VENV_PACKAGES%

set VENV_PACKAGES=

echo:

if [%2]==[] (
     set PYRL_EXTRAS=tested
 ) else ( 
     set PYRL_EXTRAS=%2
 )
 
echo Installing PyRL package in python as a local editable package...
REM REM pip install -e .
REM REM %CONDA_PATH%\envs\%VENV_NAME%\bin\python -m pip install -e %cd%
REM call %CONDA_PATH%\envs\%VENV_NAME%\Scripts\pip3.exe install -e .[%PYRL_EXTRAS%] --no-warn-script-location
REM REM %CONDA_PATH%\Scripts\conda develop . -n %VENV_NAME%
REM REM conda develop . -n %VENV_NAME%

set PYRL_EXTRAS=

REM echo - Showing existent environments
REM 
REM pip list

set PYRL_PATH=
set PATH=%OLD_PATH%
set OLD_PATH=

echo:

echo For activating this environment, please use: 
echo   venv_activate %VENV_PYRL_NAME%
echo or even:
echo   conda activate %VENV_PYRL_NAME%
echo:
echo For deactivating it, please use: 
echo   conda deactivate
echo:
echo For deleting it, please use: 
echo   venv_remove %VENV_PYRL_NAME%
echo or even:
echo   conda remove %VENV_PYRL_NAME%
echo:

echo Operation terminated.
pause
echo:

REM echo - Activating environment

REM source activate pyrl
REM %windir%\System32\cmd.exe "/C" %CONDA_PATH%\Scripts\activate.bat %CONDA_PATH%\envs\%VENV_NAME%
REM cmd /C %CONDA_PATH%\Scripts\activate.bat %CONDA_PATH%\envs\%VENV_NAME%
REM cmd /K venv_activate %VENV_PYRL_NAME%
REM cmd /K conda activate %VENV_PYRL_NAME%
REM call conda activate %VENV_PYRL_NAME%


:NOT_EXECUTED

REM set /p %VENV_PYRL_NAME%=please type test or dev:
REM if not [%VENV_PYRL_NAME%]==[test] if not [%VENV_OPTION%]==[dev] (
REM    echo Invalid option %VENV_OPTION%. Setting to "test".
REM  %VENV_OPTION%=test
REM )

REM if exist yourfoldername\ (
REM   echo Yes 
REM ) else (
REM   echo No
REM )
REM 
REM if exist yourfilename (
REM   echo Yes 
REM ) else (
REM   echo No
REM )

REM echo - Creating new environment at folder "venv"
REM 
REM c:\Anaconda3\Scripts\conda create --prefix ./venv

REM rd /S venv
REM md venv
REM virtualenv venv

REM echo - Copying necessary dll files
REM 
REM copy c:\Anaconda3\Library\bin\libcrypto-1_1-x64.dll .\venv\Scripts\ 
REM copy c:\Anaconda3\Library\bin\libcrypto-1_1-x64.pdb .\venv\Scripts\ 
REM copy c:\Anaconda3\Library\bin\libssl-1_1-x64.dll .\venv\Scripts\ 
REM copy c:\Anaconda3\Library\bin\openssl.exe .\venv\Scripts\ 
REM copy c:\Anaconda3\Library\bin\libssl-1_1-x64.pdb .\venv\Scripts\ 
REM copy c:\Anaconda3\Library\bin\openssl.pdb .\venv\Scripts\ 

REM *** IF SSL ERROR, please copy 
REM  - libcrypto-1_1-x64.*
REM  - libssl-1_1-x64.*
REM from CONDA_PATH\Library\bin to CONDA_PATH\DLLs
