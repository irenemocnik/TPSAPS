# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 19:39:45 2025

@author: iremo
"""

import numpy as np
import matplotlib.pyplot as plt

#%% Función para generar la señal seno
def seno(N, R, SNR):
    
    fs=1000
    df= fs/N
    ts = 1/fs
    w0 = fs/4
    a1=np.sqrt(2)

    t = np.arange(0,1,1/N).reshape(N, 1)
    tt=np.tile(t, (1,R))#repite tt 200 veces en el eje de las columnas

    fr = np.random.uniform(-1/2, 1/2, size=(1, R))
    w1 = w0 + fr * df

    S = a1*np.sin(w1*tt) #1000 * 200
    
    potseñal = np.mean(S**2)
    potruido = potseñal / (10**(SNR/10))
    

    N=np.random.normal(0, np.sqrt(potruido), size=S.shape)
    
    Xr = S + N
    
    return Xr, w1

Xr, w1 = seno(N=1000, R=200, SNR=10)
Xr_fft= np.fft.ftt(Xr[:,0])#la primera realizacion
XrMAG=np.abs(Xr_fft)


    
   
