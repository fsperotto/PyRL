@echo off

if not [%CONDA_DEFAULT_ENV%]==[(pyrl_dev)] (
  echo You should use "make" command while into *pyrl_dev* virtual environment. It seems not be the case.
  set /p CONTINUE_RUNNING_MAKE=Continue running anyway? [y/N]
  if not [%CONTINUE_RUNNING_MAKE%]==[y] (
    exit /b
  )
)

echo:
echo BUILD PACKAGE...
python -m build

echo:
echo BUILD DOCS...
echo Calling sphinx-apidoc...
sphinx-apidoc -f -o .\docs\source .\src\pyrl --module-first --private --implicit-namespaces

echo Calling sphinx-build...
call sphinx-build -M clean .\docs\source .\docs\build
call sphinx-build -M html .\docs\source .\docs\build 

REM echo:
REM echo INSTALL PYRL AS EDITABLE PACKAGE...
REM pip install -e .[all]
REM 
REM echo:
REM echo EXECUTE TESTS...
REM 
REM for %%f in (.\tests\*.py) do call python %%f
REM 
REM echo:

pause