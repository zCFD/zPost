#!/usr/bin/env python

import sys

data_host=sys.argv[2]
data_dir='.'
job_ntasks=sys.argv[4]
paraview_cmd='PATH=$PATH:'+ sys.argv[3] + ' /opt/vizstack/bin/viz-paraview -r ' + str(job_ntasks)

from zutil.post import pvserver_process

pvserver_process(data_host=data_host,data_dir=data_dir,paraview_cmd=paraview_cmd,job_ntasks=job_ntasks,vizstack=True)
