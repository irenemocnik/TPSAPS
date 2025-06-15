# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 15:04:13 2025

@author: iremo
"""
import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.ndimage import median_filter



fs_ecg = 1000 # Hz
sio.whosmat('ecg.mat')
mat_struct = sio.loadmat('ecg.mat')

ecg_one_lead =(mat_struct['ecg_lead'])
ecg_one_lead = ecg_one_lead.flatten()
N = len(ecg_one_lead)

nyq_frec = fs_ecg/2

w200 = int(0.2 * fs_ecg)
w600 = int(0.6 * fs_ecg)

filtroM = median_filter(ecg_one_lead, size = w200, mode = 'reflect')
b = median_filter(filtroM, size = w600, mode = 'reflect')

ecgFiltrada = ecg_one_lead - b #elimina el ruido de linea de base

#graficos de ecg, ecg filtrada
plt.figure(figsize=(12, 6))
plt.plot(ecg_one_lead, label='ECG Original')
plt.plot(ecgFiltrada, label='ECG Filtrada', color='orange')
plt.title('ECG Original')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Amplitud')
plt.grid()
plt.legend()

regs_interes = (
        np.array([5, 5.2]) *60*fs_ecg, # minutos a muestras
        np.array([12, 12.4]) *60*fs_ecg, # minutos a muestras
        np.array([15, 15.2]) *60*fs_ecg, # minutos a muestras
        )

for ii in regs_interes:
    # Intervalo acotado dentro del rango de datos
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([N, ii[1]]), dtype='uint')

    plt.figure()
   
    plt.title(f'Filtrado en región sin interferencias: muestras {int(ii[0])} a {int(ii[1])}')
    plt.ylabel('Amplitud')
    plt.xlabel('Muestras (#)')
    plt.legend()
    plt.yticks([])
    plt.show()