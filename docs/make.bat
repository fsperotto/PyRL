@ECHO OFF

REM pushd %~dp0

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
%SPHINXAPIDOC% -f -o %SOURCEDIR% %PACKAGEDIR% --module-first --private --implicit-namespaces

echo Calling sphinx-build...
if "%1" == "" (
	%SPHINXBUILD% -M clean %SOURCEDIR% %BUILDDIR%
	%SPHINXBUILD% -M html %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
) else (
	%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
)


:end

REM popd

REM pause