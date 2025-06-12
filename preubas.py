# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 17:47:28 2025

@author: iremo
"""

import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
   
import scipy.io as sio
from scipy.io.wavfile import write
from pytc2.sistemas_lineales import plot_plantilla


def vertical_flaten(a):

    return a.reshape(a.shape[0],1)

fs_ecg = 1000 # Hz
sio.whosmat('ecg.mat')
mat_struct = sio.loadmat('ecg.mat')
ecg_one_lead = vertical_flaten(mat_struct['ecg_lead'])
N = len(ecg_one_lead)
nyq_frec = fs_ecg/2
ripple = 1 # dB
atenuacion = 40 # dB

ws1 = 0.1 # Hz
wp1 = 1.0 # Hz
wp2 = 35 # Hz
ws2 = 50 # Hz
frecs = np.array([0.0,         ws1,         wp1,     wp2,     ws2,         nyq_frec   ]) / nyq_frec
gains = np.array([-atenuacion, -atenuacion, -ripple, -ripple, -atenuacion, -atenuacion])

gains = 10**(gains/20)
band = [1, 35]  
trans_width = 0.6   # Width of transition from pass to stop, Hz
numtaps = 4501         # Size of the FIR filter.
edges = [0, band[0] - trans_width, band[0], band[1], band[1] + trans_width , 0.5*fs_ecg]


num_remez = sig.remez(numtaps, edges, [0.01, 1, 0.007], fs=fs_ecg) # (20⋅log(0.01)=-40dB

den = 1.0

#Análisis:
    
NN = 1024
w_rad  = np.append(np.logspace(-2, 0.8, NN//4), np.logspace(0.9, 1.6, NN//4) )
w_rad  = np.append(w_rad, np.linspace(40, nyq_frec, NN//2, endpoint=True) ) / nyq_frec * np.pi

w, hh_remez = sig.freqz(num_remez, den, worN=w_rad)


w = w / np.pi * nyq_frec  

plt.figure(1)
plt.title(' Parks-McClellan-Remez - Respuesta en frecuencia, variación del orden')
plt.plot(w, 20 * np.log10(abs(hh_remez)))

plt.xlabel('Frequencia [Hz]')
plt.ylabel('Modulo [dB]')
plt.gca().legend()
plt.grid()
plt.axis([0, 60, -60, 1 ]);
plot_plantilla(filter_type = 'bandpass', fpass = frecs[[2, 3]]* nyq_frec, ripple = ripple , fstop = frecs[ [1, 4] ]* nyq_frec, attenuation = atenuacion, fs = fs_ecg)
plt.figure(2)
plt.title(' Parks-McClellan-Remez - Respuesta en frecuencia, variación del orden')
plt.plot(w, 20 * np.log10(abs(hh_remez)))
plt.xlabel('Frequencia [Hz]')
plt.ylabel('Modulo [dB]')
plt.gca().legend()
plt.grid()
plt.axis([0 , 1, -60, 1 ]);
plot_plantilla(filter_type = 'bandpass', fpass = frecs[[2, 3]]* nyq_frec, ripple = ripple , fstop = frecs[ [1, 4] ]* nyq_frec, attenuation = atenuacion, fs = fs_ecg)

plt.figure(3)
plt.title(' Parks-McClellan-Remez - Respuesta en frecuencia, variación del orden')
plt.plot(w, 20 * np.log10(abs(hh_remez)))
plt.xlabel('Frequencia [Hz]')
plt.ylabel('Modulo [dB]')
plt.gca().legend()
plt.grid()
plt.axis([34,38, -60, 1 ]);
plot_plantilla(filter_type = 'bandpass', fpass = frecs[[2, 3]]* nyq_frec, ripple = ripple , fstop = frecs[ [1, 4] ]* nyq_frec, attenuation = atenuacion, fs = fs_ecg)
