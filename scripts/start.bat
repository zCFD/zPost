@ECHO OFF
echo "zPost Starting IPython notebook"

..\zpost-py27\bin\activate

REM PARAVIEW_HOME="/Applications/paraview.app"

REM set DYLD_FRAMEWORK_PATH=%PARAVIEW_HOME%\Contents\Frameworks
REM set DYLD_LIBRARY_PATH=/System/Library/Frameworks/ImageIO.framework/Versions/A/Resources:%PARAVIEW_HOME%\Contents\Libraries
REM set PYTHONPATH=%PARAVIEW_HOME%\Contents\Python:%PARAVIEW_HOME%\Contents\Libraries:`pwd`/../python

cd ..\ipynb
ipython notebook
