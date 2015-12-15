@ECHO OFF
echo "zPost IPython installer"

IF NOT "%1"=="" GOTO ArgOk
echo "Supply full path for paraview executables" 
goto EOF

:ArgOk

echo "Using ParaView from: %*\pvpython.exe"

pushd "%~dp0"
pushd ..

echo "Checking for pvpython.exe"

IF EXIST "%*\pvpython.exe" goto PVpythonOK
echo "ERROR: pvpython not found"
goto EOF

:PVpythonOK

echo "Using %*\pvpython.exe"

echo "Checking for Python"
REM Check for python 2.7 or >3.3
FOR /F "tokens=1,2" %%G IN ('"python.exe -V 2>&1"') DO ECHO %%H | find "2.7" > Nul
IF NOT ErrorLevel 1 GOTO PythonOK 
FOR /F "tokens=1,2" %%G IN ('"python.exe -V 2>&1"') DO ECHO %%H | find "3.3" > Nul
IF NOT ErrorLevel 1 GOTO PythonOK 
FOR /F "tokens=1,2" %%G IN ('"python.exe -V 2>&1"') DO ECHO %%H | find "3.4" > Nul
IF NOT ErrorLevel 1 GOTO PythonOK 
ECHO Requires Python 2.7 or > 3.3
GOTO EOF

:PythonOK 
echo "Checking for virtualenv"

REM Check for virtualenv
WHERE virtualenv --version 2>NUL
IF %ERRORLEVEL% == 0 GOTO VirtualEnvOK
ECHO virtualenv not found
GOTO EOF
:VirtualEnvOK

if exist ".\zpost-py27\" rd /q /s ".\zpost-py27\"

echo "Creating virtual environment"
virtualenv --system-site-packages zpost-py27

echo "Activating virtual environment"
call zpost-py27\scripts\activate

echo "Installing yolk"
pip install yolk

echo "Installing requirements"
pip install -r requirements.txt 

yolk -l

echo "Saving Paraview site location info"
set sitelib=""
for /f "tokens=*" %%G in ('dir /b /a:d "%*"\..\lib\*') do (
  set sitelib=%%G
)

if not %sitelib% == "" goto sitelibok 
echo Paraview Sitelib not found!
goto EOF

:sitelibok
echo SET PARAVIEW_SITE_LIB=%*\..\lib\%sitelib%> zpost-py27\pv-location.bat
echo SET PARAVIEW_BIN=%* >> zpost-py27\pv-location.bat

popd

:EOF