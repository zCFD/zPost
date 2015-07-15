@ECHO OFF
echo "zPost Starting IPython notebook"

call ..\zpost-py27\Scripts\activate

REM PARAVIEW_HOME="/Applications/paraview.app"

REM set DYLD_FRAMEWORK_PATH=%PARAVIEW_HOME%\Contents\Frameworks
REM set DYLD_LIBRARY_PATH=/System/Library/Frameworks/ImageIO.framework/Versions/A/Resources:%PARAVIEW_HOME%\Contents\Libraries
REM set PYTHONPATH=%PARAVIEW_HOME%\Contents\Python:%PARAVIEW_HOME%\Contents\Libraries:`pwd`/../python

set PARAVIEW_HOME=C:\PROGRA~2\ParaView 4.3.1
set PYTHONPATH=%PARAVIEW_HOME%\lib\paraview-4.3\site-packages\;%PARAVIEW_HOME%\lib\paraview-4.3\site-packages\vtk;%CD%\..\python
set PATH=%CD%\..\zpost-py27\Scripts;%PATH%;%PARAVIEW_HOME%\bin\;%PARAVIEW_HOME%\lib\site-packages\vtk\

echo %PYTHONPATH% 
echo %PATH%

cd ..\ipynb
ipython notebook
