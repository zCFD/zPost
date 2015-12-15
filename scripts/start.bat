@ECHO OFF
echo "zPost Starting IPython notebook"

call ..\zpost-py27\Scripts\activate

call ..\pvconnect-py27\pv-location.bat

echo "%PARAVIEW_BIN_LOCATION%"
set PATH=%PARAVIEW_BIN_LOCATION%;%PATH%
set PYTHONPATH=..;%PARAVIEW_SITE_LIB%\site-packages;%PARAVIEW_SITE_LIB%\site-packages\vtk;%PYTHONPATH%

set PATH=%CD%\..\zpost-py27\Scripts;%PATH%

echo %PYTHONPATH% 
echo %PATH%

cd ..\ipynb
ipython notebook
