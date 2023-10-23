@echo off

REM DOWNLOAD FROM SOURCE (for windows)
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o miniconda.exe

REM INSTALL IT
start /wait "" miniconda.exe /S

REM DELETE INSTALL FILE
del miniconda.exe