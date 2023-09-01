@echo off

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

echo:
echo INSTALL AS EDITABLE PACKAGE...
pip install -e .

echo:
echo EXECUTE TESTS...

for %%f in (.\tests\*.py) do call python %%f

echo: