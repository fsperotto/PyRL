@echo off
echo:

echo Installing PyRL package in python as a local editable package...

pip3 install -e .[tested]
REM conda develop .[tested]
REM conda install --use-local .[tested]

REM cmd /c install_atari_roms.bat