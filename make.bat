@echo off

echo build package...
python -m build

echo build docs...
docs/make.bat
