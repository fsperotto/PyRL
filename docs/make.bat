@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)

if "%SPHINXAPIDOC%" == "" (
	set SPHINXAPIDOC=sphinx-apidoc
)

set SOURCEDIR=source
set BUILDDIR=build
set PACKAGEDIR=../src/pyrl

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.https://www.sphinx-doc.org/
	exit /b 1
)

echo Calling sphinx-apidoc...
%SPHINXAPIDOC% -o %SOURCEDIR% %PACKAGEDIR%

set MODE=%1

if "%1" == "" (
REM	%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
	set	MODE=html
)

echo Calling sphinx-build...
%SPHINXBUILD% -M %MODE% %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

:end

popd

pause