# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 23:03:53 2025

@author: iremo
"""

import numpy as np
import matplotlib.pyplot as plt

r=5.00
c=0.000002
l=0.5
wf=300.00

def funciona(r,c,l):
    w = np.arange(start = 0, stop = wf, step = 20)
    Ta = w*c*r/(np.sqrt(w*w*r*r*c*c+(1-l*c*w*w)*(1-l*c*w*w)))
    return w, Ta
w, Ta =funciona(r, c, l)
plt.figure(1)
plt.plot(w, 20*np.log10(Ta), label="Magnitud")

def funcionb(r,c,l):
    w = np.arange(start = 0, stop = wf, step = 20)
    Tb = w*c*r*l/(np.sqrt(w*w*(c*r*l + l)*(c*r*l + l) + r*r))
    return w, Tb
w, Tb =funcionb(r, c, l)
plt.figure(2)
plt.plot(w, 20*np.log10(Tb), label="Magnitud")




