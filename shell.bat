:INIT

@echo off

REM set CONDA_PATH=C:\Anaconda3
set CONDA_PATH=D:\fperotto\miniconda3

REM set VENV_NAME=pyrl

REM set ANACONDA_ADD_PATH=%CONDA_PATH%;%CONDA_PATH%\Library\mingw-w64\bin;%CONDA_PATH%\Library\usr\bin;%CONDA_PATH%\Library\bin;%CONDA_PATH%\Scripts;%CONDA_PATH%\bin;%CONDA_PATH%\condabin
REM set PATH=%ANACONDA_ADD_PATH%;%PATH%

:MAIN

REM echo Anaconda has been added to PATH
echo:
echo:
echo * special batch commands do be launched out of the virtual environment:
echo:
echo  - venv_create [dev/test]  [create the "pyrl" virtual environment, if it not exists]
echo  - venv_activate [activate the "pyrl" virtual environment, if it exists]
echo  - venv_remove [delete the "pyrl" virtual environment, if it exists]
echo  - make [(re-)make this local "pyrl" package using setuptools and sphinx]
echo:
echo * special batch commands do be launched into the virtual environment:
echo:
echo  - install [install this local "pyrl" package as an editable module]
echo  - venv_deactivate [deactivate the "pyrl" virtual environment, if activated]
echo  - run_tests [run the python scripts inside the "tests" folder]
echo:
echo:

REM cmd

REM %windir%\System32\cmd.exe "/K" %CONDA_PATH%\Scripts\activate.bat %CONDA_PATH%
REM cmd /K %CONDA_PATH%\Scripts\conda.bat activate
cmd /K %CONDA_PATH%\Scripts\activate.bat %CONDA_PATH%
