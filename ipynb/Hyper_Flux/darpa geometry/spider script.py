# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 10:35:29 2014

@author: andrei
"""


#%pylab inline
import pylab as pl

import numpy as np 
import matplotlib.pyplot as plt
import math
rmax = -0.8333333 # maximum radius of the suboff
xb = 3.333333     # bow
xm = 10.645833    # parallel plate 
xa = 13.979167    # afterbody  
xc = 14.291667    # after cap
cb1 = 1.126395101 
cb2 = 0.442874707
cb3 = 1.0/2.1
rh = 0.1175
k0 = 10.0
k1 = 44.6244
iDx_offset = 0.01          # offset 
idx = 300                  # number of points
idy = 300                  # number of points

x_le_bow = 0.0
x_tr_bow = 3.333333

x_bow = np.zeros(idx + 1)
y_bow = np.zeros(idx + 1)
a_bow = np.zeros(idx + 1)
b_bow = np.zeros(idx + 1)
x_bow[0] = 0.0
a_bow[0] = -1.0
b_bow[0] = 1.0
for i in range(1,idx + 1):
    x_bow[i] = x_bow[i-1] + x_tr_bow/idx
    a_bow[i] = 0.3 * x_bow[i] - 1.0
    b_bow[i] = 1.2 * x_bow[i] + 1.0
    if 0.0<=x_bow[i]<=3.333333:
        y_bow[i] = rmax*(cb1 * x_bow[i] * a_bow[i]**3.0 + cb2 * x_bow[i]**2.0 * a_bow[i]**3.0 + 1.0 - (a_bow[i]**4 * b_bow[i]))**(1.0/2.1)
fig = plt.figure(figsize=(20, 3),dpi=1200, facecolor='w', edgecolor='k')
plt.plot(-y_bow,'x',linestyle='-',linewidth = '2',marker='o',markersize=5)
plt.show()
       