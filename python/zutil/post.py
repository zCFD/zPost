""" Helper functions for accessing Paraview functionality
.. moduleauthor:: Zenotech Ltd
"""
from paraview.simple import *
# from paraview.vtk.util import numpy_support
try:
    from paraview.vtk.dataset_adapter import numpyTovtkDataArray
    from paraview.vtk.dataset_adapter import Table
    from paraview.vtk.dataset_adapter import PolyData
    from paraview.vtk.dataset_adapter import DataSetAttributes
    from paraview.vtk.dataset_adapter import DataSet
    from paraview.vtk.dataset_adapter import CompositeDataSet
    from paraview.vtk.dataset_adapter import PointSet
except:
    from paraview.vtk.numpy_interface.dataset_adapter import numpyTovtkDataArray
    from paraview.vtk.numpy_interface.dataset_adapter import Table
    from paraview.vtk.numpy_interface.dataset_adapter import PolyData
    from paraview.vtk.numpy_interface.dataset_adapter import DataSetAttributes
    from paraview.vtk.numpy_interface.dataset_adapter import DataSet
    from paraview.vtk.numpy_interface.dataset_adapter import CompositeDataSet
    from paraview.vtk.numpy_interface.dataset_adapter import PointSet

import pylab as pl
from zutil import rotate_vector
import json
from zutil import mag
import math
import time


def sum_and_zone_filter_array(input, array_name, ignore_zone, filter=None):
    sum = [0.0, 0.0, 0.0]
    p = input.GetCellData().GetArray(array_name)
    z = input.GetCellData().GetArray("zone")
    numCells = input.GetNumberOfCells()
    for x in range(numCells):
        if len(ignore_zone) == 0:
            v = p.GetTuple(x)
            for i in range(0, 3):
                sum[i] += v[i]
        else:
            zone = z.GetValue(x)
            if zone not in ignore_zone:
                v = p.GetTuple(x)
                if filter is None or filter.test(input, x):
                    # print 'Zone: %i'%(zone)
                    for i in range(0, 3):
                        sum[i] += v[i]
    return sum


def sum_and_zone_filter(input, array_name, ignore_zone, filter=None):
    sum = [0.0, 0.0, 0.0]
    if input.IsA("vtkMultiBlockDataSet"):
        iter = input.NewIterator()
        iter.UnRegister(None)
        iter.InitTraversal()
        while not iter.IsDoneWithTraversal():
            cur_input = iter.GetCurrentDataObject()
            v = sum_and_zone_filter_array(cur_input, array_name,
                                          ignore_zone, filter)
            for i in range(0, 3):
                sum[i] += v[i]
            iter.GoToNextItem()
    else:
        sum = sum_and_zone_filter_array(input, array_name, ignore_zone, filter)

    return sum


class GeomFilterLT:
    def __init__(self, val, idx):
        #
        self.val = val
        self.idx = idx

    def test(self, input, x):
        centre = input.GetCellData().GetArray("centre").GetTuple(x)
        if centre[self.idx] < self.val:
            return True
        else:
            return False


class GeomFilterGT:
    def __init__(self, val, idx):
        #
        self.val = val
        self.idx = idx

    def test(self, input, x):
        centre = input.GetCellData().GetArray("centre").GetTuple(x)
        if centre[self.idx] >= self.val:
            return True
        else:
            return False


def calc_force_from_file(file_name, ignore_zone, half_model=False,
                         filter=None, **kwargs):
    """ Calculates the pressure and friction force

    This function requires that the VTK file contains three cell data arrays
    called pressureforce, frictionforce and zone

    Args:
        file_name (str): the VTK file name including path
        ignore_zone (list): List of zones to be ignored

    Kwargs:
        half_nodel (bool): Does the data represent only half of the model
        filter (function):

    Returns:
        float, float. pressure force and friction force
    """
    wall = PVDReader(FileName=file_name)
    wall.UpdatePipeline()

    return calc_force(wall, ignore_zone, half_model, filter, kwargs)


def calc_force_wall(file_root, ignore_zone, half_model=False,
                    filter=None, **kwargs):

    wall = PVDReader(FileName=file_root+'_wall.pvd')
    wall.UpdatePipeline()

    return calc_force(wall, ignore_zone, half_model, filter, **kwargs)


def calc_force(surface_data, ignore_zone, half_model=False,
               filter=None, **kwargs):

    alpha = 0.0
    if 'alpha' in kwargs:
        alpha = kwargs['alpha']
    beta = 0.0
    if 'beta' in kwargs:
        beta = kwargs['beta']

    sum_client = servermanager.Fetch(surface_data)

    pforce = sum_and_zone_filter(sum_client, "pressureforce",
                                 ignore_zone, filter)
    fforce = sum_and_zone_filter(sum_client, "frictionforce",
                                 ignore_zone, filter)

    pforce = rotate_vector(pforce, alpha, beta)
    fforce = rotate_vector(fforce, alpha, beta)

    if half_model:
        for i in range(0, 3):
            pforce[i] *= 2.0
            fforce[i] *= 2.0

    return pforce, fforce


def get_span(wall):
    """ Returns the min and max y ordinate

    Args:
        wall (vtkMultiBlockDataSet): The input surface

    Returns:
        (float,float). Min y, Max y
    """
    Calculator1 = Calculator(Input=wall)

    Calculator1.AttributeMode = 'Point Data'
    Calculator1.Function = 'coords.jHat'
    Calculator1.ResultArrayName = 'ypos'
    Calculator1.UpdatePipeline()

    ymin = MinMax(Input=Calculator1)
    ymin.Operation = "MIN"
    ymin.UpdatePipeline()

    ymin_client = servermanager.Fetch(ymin)

    min_pos = ymin_client.GetPointData().GetArray("ypos").GetValue(0)

    ymax = MinMax(Input=Calculator1)
    ymax.Operation = "MAX"
    ymax.UpdatePipeline()

    ymax_client = servermanager.Fetch(ymax)

    max_pos = ymax_client.GetPointData().GetArray("ypos").GetValue(0)

    Delete(ymin)
    Delete(ymax)
    Delete(Calculator1)

    return [min_pos, max_pos]


def get_chord(slice, rotate_geometry=[0.0, 0.0, 0.0]):
    """ Returns the min and max x ordinate

    Args:
        wall (vtkMultiBlockDataSet): The input surface

    Returns:
        (float,float). Min x, Max x
    """

    transform = Transform(Input=slice, Transform="Transform")
    transform.Transform.Scale = [1.0, 1.0, 1.0]
    transform.Transform.Translate = [0.0, 0.0, 0.0]
    transform.Transform.Rotate = rotate_geometry
    transform.UpdatePipeline()

    Calculator1 = Calculator(Input=transform)

    Calculator1.AttributeMode = 'Point Data'
    Calculator1.Function = 'coords.iHat'
    Calculator1.ResultArrayName = 'xpos'
    Calculator1.UpdatePipeline()

    xmin = MinMax(Input=Calculator1)
    xmin.Operation = "MIN"
    xmin.UpdatePipeline()

    xmin_client = servermanager.Fetch(xmin)

    min_pos = xmin_client.GetPointData().GetArray("xpos").GetValue(0)

    xmax = MinMax(Input=Calculator1)
    xmax.Operation = "MAX"
    xmax.UpdatePipeline()

    xmax_client = servermanager.Fetch(xmax)

    max_pos = xmax_client.GetPointData().GetArray("xpos").GetValue(0)

    Delete(xmin)
    Delete(xmax)
    Delete(Calculator1)
    Delete(transform)

    return [min_pos, max_pos]


def get_chord_spanwise(slice):

    Calculator1 = Calculator(Input=slice)

    Calculator1.AttributeMode = 'Point Data'
    Calculator1.Function = 'coords.jHat'
    Calculator1.ResultArrayName = 'ypos'
    Calculator1.UpdatePipeline()

    ymin = MinMax(Input=Calculator1)
    ymin.Operation = "MIN"
    ymin.UpdatePipeline()

    ymin_client = servermanager.Fetch(ymin)

    min_pos = ymin_client.GetPointData().GetArray("ypos").GetValue(0)

    ymax = MinMax(Input=Calculator1)
    ymax.Operation = "MAX"
    ymax.UpdatePipeline()

    ymax_client = servermanager.Fetch(ymax)

    max_pos = ymax_client.GetPointData().GetArray("ypos").GetValue(0)

    Delete(ymin)
    Delete(ymax)
    Delete(Calculator1)

    return [min_pos, max_pos]

def get_monitor_data(file, monitor_name, var_name):
    """ Return the _report file data corresponding to a monitor point and variable name
        """
    monitor = CSVReader(FileName=[file])
    monitor.HaveHeaders = 1
    monitor.MergeConsecutiveDelimiters = 1
    monitor.UseStringDelimiter = 0
    monitor.DetectNumericColumns = 1
    monitor.FieldDelimiterCharacters = ' '
    monitor.UpdatePipeline()
    monitor_client = servermanager.Fetch(monitor)
    table = Table(monitor_client)
    data = table.RowData
    names = data.keys()
    num_var = len(names)-2
    if (str(monitor_name) + "_" + str(var_name) in names):
        index = names.index(str(monitor_name) + "_" + str(var_name))
        return (data[names[0]],data[names[index]])
    else:
        print 'POST.PY: MONITOR POINT: ' + str(monitor_name) + "_" + str(var_name) + ' NOT FOUND'


def residual_plot(file):
    """ Plot the _report file
    """
    l2norm = CSVReader(FileName=[file])
    l2norm.HaveHeaders = 1
    l2norm.MergeConsecutiveDelimiters = 1
    l2norm.UseStringDelimiter = 0
    l2norm.DetectNumericColumns = 1
    l2norm.FieldDelimiterCharacters = ' '
    l2norm.UpdatePipeline()

    l2norm_client = servermanager.Fetch(l2norm)

    table = Table(l2norm_client)

    data = table.RowData

    names = data.keys()

    num_var = len(names)-2
    num_rows = ((num_var-1)/4)+1

    fig = pl.figure(figsize=(40, 10*num_rows), dpi=100,
                    facecolor='w', edgecolor='k')

    fig.suptitle(file, fontsize=40, fontweight='bold')

    for i in range(1, num_var+1):
        var_name = names[i]
        ax = fig.add_subplot(num_rows, 4, i)
        if 'rho' in var_name:
            ax.set_yscale('log')
            ax.set_ylabel('l2norm '+var_name, multialignment='center')
        else:
            ax.set_ylabel(var_name, multialignment='center')

        ax.grid(True)
        ax.set_xlabel('Cycles')

        ax.plot(data[names[0]], data[names[i]], color='r', label=names[i])


def for_each(surface, func, **kwargs):
    if surface.IsA("vtkMultiBlockDataSet"):
        iter = surface.NewIterator()
        iter.UnRegister(None)
        iter.InitTraversal()
        while not iter.IsDoneWithTraversal():
            cur_input = iter.GetCurrentDataObject()
            # numCells = cur_input.GetNumberOfCells()
            numPts = cur_input.GetNumberOfPoints()
            if numPts > 0:
                calc = DataSet(cur_input)
                pts = PointSet(cur_input)
                func(calc, pts, **kwargs)
            iter.GoToNextItem()
    else:
        calc = DataSet(surface)
        pts = PointSet(surface)
        func(calc, pts, **kwargs)


def cp_profile_wall_from_file(file_root, slice_normal,
                              slice_origin, **kwargs):

    wall = PVDReader(FileName=file_root+'_wall.pvd')
    clean = CleantoGrid(Input=wall)
    clean.UpdatePipeline()
    merged = MergeBlocks(Input=clean)
    merged.UpdatePipeline()
    return cp_profile(merged, slice_normal, slice_origin, **kwargs)


def cp_profile_wall_from_file_span(file_root, slice_normal,
                                   slice_origin, **kwargs):

    wall = PVDReader(FileName=file_root+'_wall.pvd')
    clean = CleantoGrid(Input=wall)
    clean.UpdatePipeline()
    merged = MergeBlocks(Input=clean)
    merged.UpdatePipeline()
    return cp_profile_span(merged, slice_normal, slice_origin, **kwargs)


def cp_profile(surface, slice_normal, slice_origin, **kwargs):

    alpha = 0.0
    if 'alpha' in kwargs:
        alpha = kwargs['alpha']
    beta = 0.0
    if 'beta' in kwargs:
        beta = kwargs['beta']

    time_average = False
    if 'time_average' in kwargs:
        time_average = kwargs['time_average']

    rotate_geometry = [0.0, 0.0, 0.0]
    if 'rotate_geometry' in kwargs:
        rotate_geometry = kwargs['rotate_geometry']

    point_data = CellDatatoPointData(Input=surface)
    point_data.PassCellData = 1

    slice = Slice(Input=point_data, SliceType="Plane")

    slice.SliceType.Normal = slice_normal
    slice.SliceType.Origin = slice_origin

    slice.UpdatePipeline()

    if time_average:
        temporal = TemporalStatistics(Input=slice)
        temporal.ComputeMaximum = 0
        temporal.ComputeStandardDeviation = 0
        temporal.ComputeMinimum = 0
        temporal.UpdatePipeline()
        slice = temporal

    offset = get_chord(slice, rotate_geometry)

    transform = Transform(Input=slice, Transform="Transform")
    transform.Transform.Scale = [1.0, 1.0, 1.0]
    transform.Transform.Translate = [0.0, 0.0, 0.0]
    transform.Transform.Rotate = rotate_geometry
    transform.UpdatePipeline()

    chord_calc = Calculator(Input=transform)
    chord_calc.AttributeMode = 'Point Data'
    chord_calc.Function = ('(coords.iHat - ' + str(offset[0]) + ')/' +
                           str(offset[1]-offset[0]))
    chord_calc.ResultArrayName = 'chord'

    # Attempt to calculate forces
    pforce = [0.0, 0.0, 0.0]
    fforce = [0.0, 0.0, 0.0]

    sum = MinMax(Input=slice)
    sum.Operation = "SUM"
    sum.UpdatePipeline()

    sum_client = servermanager.Fetch(sum)
    if (sum_client.GetCellData().GetArray("pressureforce") and
            sum_client.GetCellData().GetArray("frictionforce")):

        pforce = sum_client.GetCellData().GetArray("pressureforce").GetTuple(0)
        fforce = sum_client.GetCellData().GetArray("frictionforce").GetTuple(0)

        pforce = rotate_vector(pforce, alpha, beta)
        fforce = rotate_vector(fforce, alpha, beta)
        """
        # Add sectional force integration
        sorted_line = PlotOnSortedLines(Input=chord_calc)
        sorted_line.UpdatePipeline()
        sorted_line = servermanager.Fetch(sorted_line)
        cp_array = sorted_line.GetCellData().GetArray("cp")

        for i in range(0,len(cp_array)):
            sorted_line.GetPointData().GetArray("X")
            pass
        """

    if 'func' in kwargs:
        sorted_line = PlotOnSortedLines(Input=chord_calc)
        sorted_line.UpdatePipeline()
        extract_client = servermanager.Fetch(sorted_line)
        for_each(extract_client, **kwargs)

    return {'pressure force': pforce,
            'friction force': fforce}


def cp_profile_span(surface, slice_normal, slice_origin, **kwargs):

    alpha = 0.0
    if 'alpha' in kwargs:
        alpha = kwargs['alpha']
    beta = 0.0
    if 'beta' in kwargs:
        beta = kwargs['beta']

    point_data = CellDatatoPointData(Input=surface)
    point_data.PassCellData = 1
    clip = Clip(Input=point_data, ClipType="Plane")
    clip.ClipType.Normal = [0.0, 1.0, 0.0]
    clip.ClipType.Origin = [0.0, 0.0, 0.0]
    clip.UpdatePipeline()

    slice = Slice(Input=clip, SliceType="Plane")

    slice.SliceType.Normal = slice_normal
    slice.SliceType.Origin = slice_origin

    slice.UpdatePipeline()

    offset = get_chord_spanwise(slice)
    # define the cuts and make sure the is the one one you want
    # make the
    chord_calc = Calculator(Input=slice)

    chord_calc.AttributeMode = 'Point Data'
    chord_calc.Function = ('(coords.jHat - ' + str(offset[0]) + ')/' +
                           str(offset[1]-offset[0]))
    chord_calc.ResultArrayName = 'chord'

    sum = MinMax(Input=slice)
    sum.Operation = "SUM"
    sum.UpdatePipeline()

    sum_client = servermanager.Fetch(sum)
    pforce = sum_client.GetCellData().GetArray("pressureforce").GetTuple(0)
    fforce = sum_client.GetCellData().GetArray("frictionforce").GetTuple(0)

    pforce = rotate_vector(pforce, alpha, beta)
    fforce = rotate_vector(fforce, alpha, beta)

    if 'func' in kwargs:
        sorted_line = PlotOnSortedLines(Input=chord_calc)
        sorted_line.UpdatePipeline()
        extract_client = servermanager.Fetch(sorted_line)
        for_each(extract_client, **kwargs)

    return {'pressure force': pforce,
            'friction force': fforce}


def cf_profile(surface, slice_normal, slice_origin, **kwargs):

    alpha = 0.0
    if 'alpha' in kwargs:
        alpha = kwargs['alpha']
    beta = 0.0
    if 'beta' in kwargs:
        beta = kwargs['beta']

    point_data = CellDatatoPointData(Input=surface)
    point_data.PassCellData = 1

    slice = Slice(Input=point_data, SliceType="Plane")

    slice.SliceType.Normal = slice_normal
    slice.SliceType.Origin = slice_origin

    slice.UpdatePipeline()

    offset = get_chord(slice)

    chord_calc = Calculator(Input=slice)

    chord_calc.AttributeMode = 'Point Data'
    chord_calc.Function = ('(coords.iHat - ' + str(offset[0]) + ')/' +
                           str(offset[1]-offset[0]))
    chord_calc.ResultArrayName = 'chord'

    cf_calc = Calculator(Input=chord_calc)

    cf_calc.AttributeMode = 'Point Data'
    cf_calc.Function = 'mag(cf)'
    cf_calc.ResultArrayName = 'cfmag'

    sum = MinMax(Input=slice)
    sum.Operation = "SUM"
    sum.UpdatePipeline()

    sum_client = servermanager.Fetch(sum)
    pforce = sum_client.GetCellData().GetArray("pressureforce").GetTuple(0)
    fforce = sum_client.GetCellData().GetArray("frictionforce").GetTuple(0)

    pforce = rotate_vector(pforce, alpha, beta)
    fforce = rotate_vector(fforce, alpha, beta)

    if 'func' in kwargs:
        sorted_line = PlotOnSortedLines(Input=cf_calc)
        sorted_line.UpdatePipeline()
        extract_client = servermanager.Fetch(sorted_line)
        for_each(extract_client, **kwargs)

    return {'pressure force': pforce,
            'friction force': fforce}

import csv


def get_csv_data(filename, header=False, remote=False, delim=' '):
    """ Get csv data
    """
    if remote:
        theory = CSVReader(FileName=[filename])
        theory.HaveHeaders = 0
        if header:
            theory.HaveHeaders = 1
        theory.MergeConsecutiveDelimiters = 1
        theory.UseStringDelimiter = 0
        theory.DetectNumericColumns = 1
        theory.FieldDelimiterCharacters = delim
        theory.UpdatePipeline()
        theory_client = servermanager.Fetch(theory)
        table = Table(theory_client)
        data = table.RowData
    else:
        import pandas as pd
        if not header:
            data = pd.read_csv(filename, sep=delim, header=None)
        else:
            data = pd.read_csv(filename, sep=delim)
    return data


def get_fw_csv_data(filename, widths, header=False, remote=False, **kwargs):

    if remote:
        theory = CSVReader(FileName=[filename])
        theory.HaveHeaders = 0
        theory.MergeConsecutiveDelimiters = 1
        theory.UseStringDelimiter = 0
        theory.DetectNumericColumns = 1
        theory.FieldDelimiterCharacters = ' '
        theory.UpdatePipeline()

        theory_client = servermanager.Fetch(theory)

        table = Table(theory_client)

        data = table.RowData

    else:
        import pandas as pd
        if not header:
            data = pd.read_fwf(filename, sep=' ', header=None,
                               widths=widths, **kwargs)
        else:
            data = pd.read_fwf(filename, sep=' ', width=widths, **kwargs)

    return data


def screenshot(wall):
    # position camera
    view = GetActiveView()
    if not view:
        # When using the ParaView UI, the View will be present, not otherwise.
        view = CreateRenderView()
    view.CameraViewUp = [0, 0, 1]
    view.CameraFocalPoint = [0, 0, 0]
    view.CameraViewAngle = 45
    view.CameraPosition = [5, 0, 0]

    # draw the object
    Show()

    # set the background color
    view.Background = [1, 1, 1]  # white

    # set image size
    view.ViewSize = [200, 300]  # [width, height]

    dp = GetDisplayProperties()

    # set point color
    dp.AmbientColor = [1, 0, 0]  # red

    # set surface color
    dp.DiffuseColor = [0, 1, 0]  # blue

    # set point size
    dp.PointSize = 2

    # set representation
    dp.Representation = "Surface"

    Render()

    # save screenshot
    WriteImage("test.png")


def sum_array(input, array_name):
    sum = [0.0, 0.0, 0.0]
    p = input.GetCellData().GetArray(array_name)
    numCells = input.GetNumberOfCells()
    for x in range(numCells):
        v = p.GetTuple(x)
        for i in range(0, 3):
            sum[i] += v[i]
    return sum

from fabric.api import (env, run, cd, get, hide, settings,
                        remote_tunnel, show, shell_env)
from fabric.tasks import execute


import logging
log = logging.getLogger("paramiko.transport")
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
log.addHandler(sh)

import sys
import multiprocessing as mp
from multiprocessing import Process, Value
process_id = None
use_multiprocess = True
# Uncomment for output logging
# logger = mp.get_logger()
# logger.addHandler(logging.StreamHandler(sys.stdout))
# logger.setLevel(mp.SUBDEBUG)


def pvserver(remote_dir, paraview_cmd, paraview_port, paraview_remote_port):

    with show('debug'), remote_tunnel(int(paraview_remote_port),local_port=int(paraview_port)), cd(remote_dir):
        # with cd(remote_dir):
        if not use_multiprocess:
            run('sleep 2;'+paraview_cmd+'</dev/null &>/dev/null&', pty=False)
        else:
            #    # run('sleep 2;'+paraview_cmd+'&>/dev/null',pty=False)
            run('sleep 2;'+paraview_cmd)  # , pty=False)
        # run(paraview_cmd+'</dev/null &>/dev/null',pty=False)
        # run('screen -d -m "yes"')
    # ssh asrc2 "(ls</dev/null &>/dev/null&) 2>&1; true" 2>/dev/null || echo SSH connection or remote command failed - either of them returned non-zero exit code $?


def pvcluster(remote_dir, paraview_home, paraview_args,
              paraview_port, paraview_remote_port, job_dict):

    with show('debug'), remote_tunnel(int(paraview_remote_port), local_port=int(paraview_port)):
        with shell_env(PARAVIEW_HOME=paraview_home, PARAVIEW_ARGS=paraview_args):
            run('echo $PARAVIEW_HOME')
            run('echo $PARAVIEW_ARGS')
            run('mkdir -p '+remote_dir)
            with cd(remote_dir):
                cmd_line = 'mycluster --create pvserver.job --jobname=pvserver'
                cmd_line += ' --jobqueue ' + job_dict['job_queue']
                cmd_line += ' --ntasks ' + job_dict['job_ntasks']
                cmd_line += ' --taskpernode ' + job_dict['job_ntaskpernode']
                if 'vizstack' in paraview_args:
                    cmd_line += ' --script mycluster-viz-paraview.bsh'
                else:
                    cmd_line += ' --script mycluster-paraview.bsh'
                cmd_line += ' --project ' + job_dict['job_project']
                run(cmd_line)
                run('chmod u+rx pvserver.job')
                run('mycluster --immediate --submit pvserver.job')


def port_test(rport, lport):
    # Run a test
    with hide('everything'), remote_tunnel(int(rport), local_port=int(lport)):
        run('cd')


def get_case_file():
    with cd(remote_dir):
        get(case_name+'.py', '%(path)s')


def cat_case_file(remote_dir, case_name):
    with cd(remote_dir):
        with hide('output', 'running', 'warnings'), settings(warn_only=True):
            # cmd = 'cat '+case_name+'.py'
            import StringIO
            contents = StringIO.StringIO()
            get(case_name+'.py', contents)
            # operate on 'contents' like a file object here, e.g. 'print
            return contents.getvalue()


def cat_status_file(remote_dir, case_name):
    with cd(remote_dir):
        with hide('output', 'running', 'warnings'), settings(warn_only=True):
            # cmd = 'cat '+case_name+'_status.txt'
            import StringIO
            contents = StringIO.StringIO()
            result = get(case_name+'_status.txt', contents)
            if result.succeeded:
                # operate on 'contents' like a file object here, e.g. 'print
                return contents.getvalue()
            else:
                return None


def run_uname(with_tunnel):

    with hide('everything'):
        run('uname -a')


def test_ssh(status, **kwargs):
    global data_host
    _remote_host = data_host
    if 'data_host' in kwargs:
        _remote_host = kwargs['data_host']
    try:
        env.use_ssh_config = True
        execute(run_uname, False, hosts=[_remote_host])
    except:
        status.value = 0
        return False
    return True


def test_ssh_mp(**kwargs):

    # print 'Starting test ssh'
    status = Value('i', 1)
    process_id = mp.Process(target=test_ssh, args=(status,), kwargs=kwargs)
    process_id.start()
    process_id.join()
    if status.value == 0:
        return False

    return True


def test_remote_tunnel(**kwargs):
    global data_host

    _remote_host = data_host
    if 'data_host' in kwargs:
        _remote_host = kwargs['data_host']

    try:
        env.use_ssh_config = True
        execute(run_uname, True, hosts=[_remote_host])
    except:
        return False

    return True


def get_remote_port(**kwargs):

    global data_host, paraview_remote_port, paraview_port

    _remote_host = data_host
    if 'data_host' in kwargs:
        _remote_host = kwargs['data_host']

    paraview_port = '11111'
    if 'paraview_port' in kwargs:
        paraview_port = kwargs['paraview_port']

    paraview_remote_port = '11113'
    if 'paraview_remote_port' in kwargs:
        paraview_remote_port = kwargs['paraview_remote_port']
    else:
        # Attempt to find an unused remote port
        print 'Attempting to find unused port'
        for p in range(12000, 13000):
            tp = Value('i', p)

            process_id = mp.Process(target=test_remote_port,
                                    args=(port_test, tp,
                                          paraview_port, _remote_host))
            process_id.start()
            process_id.join()
            # print tp.value
            if tp.value != 0:
                break

        print 'Selected Port: '+str(p)
        paraview_remote_port = p


def test_remote_port(port_test, port, paraview_port, remote_host):

    try:
        env.use_ssh_config = True
        execute(port_test, port.value, paraview_port, hosts=[remote_host])
        return True
    except:
        port.value = 0
        return False


def pvserver_start(remote_host, remote_dir, paraview_cmd):
    if paraview_cmd is not None:
        env.use_ssh_config = True
        execute(pvserver, remote_dir, paraview_cmd, hosts=[remote_host])


def pvserver_connect(**kwargs):
    """
    Be careful when adding to this function fabric execute calls do not play
    well with multiprocessing. Do not mix direct fabric execute call and
    mp based fabric execute calls
    """
    global remote_data, data_dir, data_host, remote_server_auto
    global paraview_cmd, process_id, paraview_port, paraview_remote_port
    global process_id

    _paraview_cmd = paraview_cmd
    if 'paraview_cmd' in kwargs:
        _paraview_cmd = kwargs['paraview_cmd']

    if '-sp' in _paraview_cmd or '--client-host' in _paraview_cmd:
        print('pvserver_process: Please only provide pvserver'
              'executable path and name without arguments')
        print 'e.g. mpiexec -n 1 /path_to_pvserver/bin/pvserver'
        return False

    # Add Check for passwordless ssh
    print 'Testing passwordless ssh access'
    if not test_ssh_mp(**kwargs):
        print 'ERROR: Passwordless ssh access to data host failed'
        return False
    print '-> Passed'

    # Add check for paraview version

    # Find free remote port
    get_remote_port(**kwargs)

    paraview_port = '11111'
    if 'paraview_port' in kwargs:
        paraview_port = kwargs['paraview_port']

    if not use_multiprocess:
        pvserver_process(**kwargs)
    else:
        print 'Starting pvserver connect'
        process_id = mp.Process(target=pvserver_process, kwargs=kwargs)
        process_id.start()
        # process_id.join()

    # time.sleep(6)

    ReverseConnect(paraview_port)

    return True


def pvcluster_process(**kwargs):
    pvserver_process(**kwargs)


def pvserver_process(**kwargs):

    global remote_data, data_dir, data_host, remote_server_auto
    global paraview_cmd, paraview_home, paraview_port, paraview_remote_port

    print 'Starting pvserver process'

    _remote_dir = data_dir
    if 'data_dir' in kwargs:
        _remote_dir = kwargs['data_dir']
    _paraview_cmd = paraview_cmd
    if 'paraview_cmd' in kwargs:
        _paraview_cmd = kwargs['paraview_cmd']
    _paraview_home = paraview_home
    if 'paraview_home' in kwargs:
        _paraview_home = kwargs['paraview_home']
    paraview_port = '11111'
    if 'paraview_port' in kwargs:
        paraview_port = kwargs['paraview_port']

    """
    _job_ntasks = 1
    if 'job_ntasks' in kwargs:
        _job_ntasks = kwargs['job_ntasks']
    """

    _remote_host = data_host
    if 'data_host' in kwargs:
        _remote_host = kwargs['data_host']

    # This global variable may have already been set so check
    if 'paraview_remote_port' not in globals():
        paraview_remote_port = '11113'
        if 'paraview_remote_port' in kwargs:
            paraview_remote_port = kwargs['paraview_remote_port']
        else:
            # Attempt to find an unused remote port
            print 'Attempting to find unused port'
            for p in range(12000, 13000):
                try:
                    env.use_ssh_config = True
                    execute(port_test, p, paraview_port, hosts=[_remote_host])
                    break
                except:
                    pass
            print 'Selected Port: '+str(p)
            paraview_remote_port = p

    if 'job_queue' in kwargs:
        # Submit job

        remote_hostname = _remote_host[_remote_host.find('@')+1:]

        if 'vizstack' in kwargs:
            paraview_args = ('/opt/vizstack/bin/viz-paraview -r ' +
                             str(kwargs['job_ntasks']) + ' -c ' +
                             remote_hostname + ' -p ' +
                             str(paraview_remote_port))
        else:
            paraview_args = (' -rc --client-host=' + remote_hostname +
                             ' -sp=' + str(paraview_remote_port))

        print paraview_args

        job_dict = {
            'job_queue': kwargs['job_queue'],
            'job_ntasks': kwargs['job_ntasks'],
            'job_ntaskpernode': kwargs['job_ntaskpernode'],
            'job_project': kwargs['job_project'],
        }
        if _paraview_home is not None:
            env.use_ssh_config = True
            execute(pvcluster, _remote_dir, _paraview_home, paraview_args,
                    paraview_port, paraview_remote_port,
                    job_dict, hosts=[_remote_host])
    else:
        # Run Paraview
        if '-sp' in _paraview_cmd or '--client-host' in _paraview_cmd:
            print ('pvserver_process: Please only provide pvserver'
                   'executable path and name without arguments')
            print 'e.g. mpiexec -n 1 /path_to_pvserver/bin/pvserver'
            return False
        if 'vizstack' in kwargs:
            _paraview_cmd = (_paraview_cmd + ' -c localhost ' + ' -p ' +
                             str(paraview_remote_port))
        else:
            _paraview_cmd = (_paraview_cmd +
                             ' -rc --client-host=localhost -sp=' +
                             str(paraview_remote_port))

        if _paraview_cmd is not None:
            env.use_ssh_config = True
            execute(pvserver, _remote_dir, _paraview_cmd, paraview_port,
                    paraview_remote_port, hosts=[_remote_host])


def pvserver_disconnect():
    Disconnect()
    if process_id:
        process_id.terminate()


def get_case_parameters(case_name, **kwargs):
    global remote_data,data_dir,data_host,remote_server_auto,paraview_cmd
    _remote_dir = data_dir
    if 'data_dir' in kwargs:
        _remote_dir = kwargs['data_dir']
    _remote_host = data_host
    if 'data_host' in kwargs:
        _remote_host = kwargs['data_host']

    env.use_ssh_config = True
    env.host_string = _remote_host
    case_file_str=cat_case_file(_remote_dir,case_name)
    exec case_file_str
    return parameters


def get_status_dict(case_name, **kwargs):
    global remote_data,data_dir,data_host,remote_server_auto,paraview_cmd
    _remote_dir = data_dir
    if 'data_dir' in kwargs:
        _remote_dir = kwargs['data_dir']
    _remote_host = data_host
    if 'data_host' in kwargs:
        _remote_host = kwargs['data_host' ]

    env.use_ssh_config = True
    env.host_string = _remote_host
    status_file_str=cat_status_file(_remote_dir,case_name)

    if status_file_str != None:
        #print status_file_str
        return json.loads(status_file_str)
    else:
        print 'WARNING: '+case_name+'_status.txt file not found'
        return None


def get_num_procs(case_name,**kwargs):#remote_host,remote_dir,case_name):
    status = get_status_dict(case_name,**kwargs)
    if 'num processor' in status:
        return status['num processor']
    else:
        return None


def get_case_root(case_name,num_procs):
    return case_name+'_P'+num_procs+'_OUTPUT/'+case_name


def get_case_report(case):
    return case+'_report.csv'


def print_html_parameters(parameters):

    reference = parameters['reference']
    material  = parameters['material']
    conditions = parameters[reference]
    
    mach = 0.0
    speed = 0.0
    
    if 'Mach' in conditions['V']:
        mach = conditions['V']['Mach']
        speed = 0.0
    else:
        speed = mag(conditions['V']['vector'])
        mach = 0.0
    
    if 'Reynolds No' in conditions:
        reynolds = conditions['Reynolds No']
    else:
        reynolds = 'undefined'
    
    if 'Reference Length' in conditions:
        reflength = conditions['Reference Length']
    else:
        reflength = 'undefined'
    
    import string
    
    html_template='''<table>
<tr><td>pressure</td><td>$pressure</td></tr>
<tr><td>temperature</td><td>$temperature</td></tr>
<tr><td>Reynolds No</td><td>$reynolds</td></tr>
<tr><td>Ref length</td><td>$reflength</td></tr>
<tr><td>Speed</td><td>$speed</td></tr>
<tr><td>Mach No</td><td>$mach</td></tr>
</table>'''
    html_output=string.Template(html_template)
    
    return html_output.substitute({'pressure':conditions['pressure'],
                        'temperature':conditions['temperature'],
                        'reynolds':reynolds,
                        'reflength':reflength,
                        'speed':speed,
                        'mach':mach,
                        })

import uuid
import time
from IPython.display import HTML, Javascript, display


class ProgressBar(object):

    def __init__(self):
        self.divid = str(uuid.uuid4())
        self.val = 0
        pb = HTML(
            """
        <div style="border: 1px solid black; width:500px">
          <div id="%s" style="background-color:grey; width:0%%">&nbsp;</div>
        </div>
        """ % self.divid)
        display(pb)

    def __iadd__(self, v):
        self.update(self.val+v)
        return self

    def complete(self):
        self.update(100)
        display(Javascript("$('div#%s').hide()" % (self.divid)))

    def update(self, i):
        self.val = i
        display(Javascript("$('div#%s').width('%i%%')" % (self.divid, i)))


remote_data = True
data_dir = 'data'
data_host = 'user@server'
remote_server_auto = True
paraview_cmd = 'mpiexec pvserver'
paraview_home = '/usr/local/bin/'
job_queue = 'default'
job_tasks = 1
job_ntaskpernode = 1
job_project = 'default'


def data_location_form_html(**kwargs):
    global remote_data, data_dir, data_host, remote_server_auto, paraview_cmd
    global job_queue, job_tasks, job_ntaskpernode, job_project

    if 'data_dir' in kwargs:
        data_dir = kwargs['data_dir']
    if 'paraview_cmd' in kwargs:
        paraview_cmd = kwargs['paraview_cmd']
    if 'data_host' in kwargs:
        data_host = kwargs['data_host']

    remote_data_checked = ''
    if remote_data:
        remote_data_checked = 'checked="checked"'
    remote_server_auto_checked = ''
    if remote_server_auto:
        remote_server_auto_checked = 'checked="checked"'

    remote_cluster_checked = ''
    job_queue = 'default'
    job_tasks = 1
    job_ntaskpernode = 1
    job_project = 'default'

    input_form = """
<div style="background-color:gainsboro; border:solid black; width:640px; padding:20px;">
<label style="width:22%;display:inline-block">Remote Data</label>
<input type="checkbox" id="remote_data" value="remote" {remote_data_checked}><br>
<label style="width:22%;display:inline-block">Data Directory</label>
<input style="width:75%;" type="text" id="data_dir" value="{data_dir}"><br>
<label style="width:22%;display:inline-block">Data Host</label>
<input style="width:75%;" type="text" id="data_host" value="{data_host}"><br>
<label style="width:22%;display:inline-block">Remote Server Auto</label>
<input type="checkbox" id="remote_server_auto" value="remote_auto" {remote_server_auto_checked}><br>
<label style="width:22%;display:inline-block">Paraview Cmd </label>
<input style="width:75%;" type="text" id="paraview_cmd" value="{paraview_cmd}"><br>
<label style="width:22%;display:inline-block">Remote Cluster</label>
<input type="checkbox" id="remote_cluster" value="remote_cluster" {remote_cluster_checked}><br>
<label style="width:22%;display:inline-block">Job Queue </label>
<input style="width:75%;" type="text" id="job_queue" value="{job_queue}"><br>
<label style="width:22%;display:inline-block">Job Tasks </label>
<input style="width:75%;" type="text" id="job_tasks" value="{job_tasks}"><br>
<label style="width:22%;display:inline-block">Job Tasks per Node </label>
<input style="width:75%;" type="text" id="job_ntaskpernode" value="{job_ntaskpernode}"><br>
<label style="width:22%;display:inline-block">Job Project </label>
<input style="width:75%;" type="text" id="job_project" value="{job_project}"><br>
<button onclick="apply()">Apply</button>
</div>
"""

    javascript = """
    <script type="text/Javascript">
        function apply(){
            var remote_data = ($('input#remote_data').is(':checked') ? 'True' : 'False');
            var data_dir = $('input#data_dir').val();
            var data_host = $('input#data_host').val();
            var remote_server_auto = ($('input#remote_server_auto').is(':checked') ? 'True' : 'False');
            var paraview_cmd = $('input#paraview_cmd').val();
            var remote_cluster = ($('input#remote_cluster').is(':checked') ? 'True' : 'False');


            var kernel = IPython.notebook.kernel;

            // Send data dir to ipython
            var  command = "from zutil import post; post.data_dir = '" + data_dir + "'";
            console.log("Executing Command: " + command);
            kernel.execute(command);

            // Send data host to ipython
            var  command = "from zutil import post; post.data_host = '" + data_host + "'";
            console.log("Executing Command: " + command);
            kernel.execute(command);

            // Send remote server flag to ipython
            var  command = "from zutil import post; post.remote_server_auto = " + remote_server_auto;
            console.log("Executing Command: " + command);
            kernel.execute(command);

            // Send paraview command to ipython
            var  command = "from zutil import post; post.paraview_cmd = '" + paraview_cmd + "'";
            console.log("Executing Command: " + command);
            kernel.execute(command);

            // Send remote data flag to ipython
            var  command = "from zutil import post; post.remote_data = " + remote_data ;
            console.log("Executing Command: " + command);
            kernel.execute(command);

            // Set paraview command to none if not using remote server
            var command = "from zutil import post; if not post.remote_server_auto: post.paraview_cmd=None"
            console.log("Executing Command: " + command);
            kernel.execute(command);

            // Set data to local host for local data
            var command = "from zutil import post; if not post.post.remote_data: post.data_host='localhost'; post.paraview_cmd=None"
            console.log("Executing Command: " + command);
            kernel.execute(command);

            if(remote_cluster == 'True'){
                // Set cluster job info
                //var command = "from zutil import post; post.jo";
            }

        }
    </script>
    """

    return HTML(input_form.format(data_dir=data_dir,
                                  data_host=data_host,
                                  paraview_cmd=paraview_cmd,
                                  remote_data_checked=remote_data_checked,
                                  remote_server_auto_checked=remote_server_auto_checked,
                                  remote_cluster_checked=remote_cluster_checked,
                                  job_queue=job_queue,
                                  job_tasks=job_tasks,
                                  job_ntaskpernode=job_ntaskpernode,
                                  job_project=job_project) + javascript)
