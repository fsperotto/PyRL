:INIT

@echo off

set CONDA_PATH=C:\Anaconda3

set VENV_NAME=pyrl

set ANACONDA_ADD_PATH=%CONDA_PATH%;%CONDA_PATH%\Library\mingw-w64\bin;%CONDA_PATH%\Library\usr\bin;%CONDA_PATH%\Library\bin;%CONDA_PATH%\Scripts;%CONDA_PATH%\bin;%CONDA_PATH%\condabin
set PATH=%PATH%;%ANACONDA_ADD_PATH%

:MAIN

echo Installing ALE ATARI ROMS

python -c "import AutoROM; AutoROM.cli()"