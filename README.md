zPost
=====

Example IPython notebooks for post processing zCFD results

Installation instructions:

Ensure you have Python (including virtualenv package) and Paraview installed. 
Note ParaView needs to use the same version of python

In the scripts folder run 

> create_virtualenv.bsh

This will install all the required python packages in zPost/zpost-py27

To run the ipython notebook server

> start_notebook

If you want to run a custom version of Paraview please set the PARAVIEW_HOME variable in your shell before starting the notebook server


Paraview Client
---------------

This library can also be used to simplify the launching of a pvserver from the ParaView client. The launcher scripts setup a secure reverse connection from the ParaView server using an automatically discovered unused port in the range of 12000-13000.

For an example ParaView server config file see shared/servers.pvsc

The launcher scripts are in scripts/pvserver_launcher.bsh and scripts/pvcluster_launcher.bsh

This relies on passwordless ssh key based authentication on the remote server so ensure this works correctly before using this facility.

Server Dependencies
^^^^^^^^^^^^^^^^^^^
The server environment requires the MyCluster application to the installed and configured correctly for the cluster capability to operate.
The server submit node also requires that the ssh server is configured to act as a gateway so that the cluster nodes can connect to the ParaView client. 

These options need to be set in the /etc/ssh/sshd_config file on the server

AllowTcpForwarding yes
GatewayPorts yes 



