@echo off

echo - Creating new environment at folder "venv"

rd /S venv
md venv
virtualenv venv

echo - Copying necessary dll files

copy c:\Anaconda3\Library\bin\libcrypto-1_1-x64.dll .\venv\Scripts\ 
copy c:\Anaconda3\Library\bin\libcrypto-1_1-x64.pdb .\venv\Scripts\ 
copy c:\Anaconda3\Library\bin\libssl-1_1-x64.dll .\venv\Scripts\ 
copy c:\Anaconda3\Library\bin\openssl.exe .\venv\Scripts\ 
copy c:\Anaconda3\Library\bin\libssl-1_1-x64.pdb .\venv\Scripts\ 
copy c:\Anaconda3\Library\bin\openssl.pdb .\venv\Scripts\ 

echo - Activating environment

venv\Scripts\activate.bat

echo - Showing existent environments

pip list

pause