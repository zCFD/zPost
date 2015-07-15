@ECHO OFF
echo "zPost IPython installer"

pushd ..

echo "Checking for Python
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

popd

:EOF