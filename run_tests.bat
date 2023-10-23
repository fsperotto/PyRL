@echo off

echo:
echo EXECUTE TESTS...

for %%f in (.\tests\*.py) do call python %%f

echo:

pause