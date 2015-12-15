@ECHO OFF
echo "zPost Starting IPython notebook"

call .\pv-location.bat

set PATH=%PARAVIEW_BIN%;%PATH%
set PYTHONPATH=..;%PARAVIEW_SITE_LIB%\site-packages;%PARAVIEW_SITE_LIB%\site-packages\vtk;%PYTHONPATH%

cd ..\ipynb
ipython notebook
