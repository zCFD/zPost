@ECHO OFF
echo "zPost IPython installer"

IF NOT "%1"=="" GOTO ArgOk
echo "Supply full path for paraview executables" 
goto EOF

:ArgOk

echo "Checking for Python"
REM Check for python 2.7 or >3.3
FOR /F "tokens=1,2" %%G IN ('"python.exe -V 2>&1"') DO ECHO %%H | find "2.7" > Nul
IF NOT ErrorLevel 1 GOTO PythonOK 
FOR /F "tokens=1,2" %%G IN ('"python.exe -V 2>&1"') DO ECHO %%H | find "3.3" > Nul
IF NOT ErrorLevel 1 GOTO PythonOK 
FOR /F "tokens=1,2" %%G IN ('"python.exe -V 2>&1"') DO ECHO %%H | find "3.4" > Nul
IF NOT ErrorLevel 1 GOTO PythonOK 
echo "Requires Python 2.7 or > 3.3, Anaconda Python distribution is recommended"
GOTO EOF

:PythonOK 

echo "Checking for Paraview"

IF EXIST "%*\pvpython.exe" goto PVpythonOK
echo "ERROR: pvpython not found"
goto EOF

:PVpythonOK

echo "Using Paraview binaries in %*"

pushd "%~dp0"
pushd ..

echo "Saving Paraview site location info"
set sitelib=""
for /f "tokens=*" %%G in ('dir /b /a:d "%*"\..\lib\*') do (
  set sitelib=%%G
)

if not %sitelib% == "" goto sitelibok 
echo Paraview Sitelib not found!
goto EOF

:sitelibok
echo SET PARAVIEW_SITE_LIB=%*\..\lib\%sitelib%> pv-location.bat
echo SET PARAVIEW_BIN=%*>> pv-location.bat

popd

:EOF