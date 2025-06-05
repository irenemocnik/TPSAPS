# -*- coding: utf-8 -*-
"""
Created on Wed May 28 20:38:50 2025

@author: iremo
"""


import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
import scipy.io as sio
from pytc2.sistemas_lineales import plot_plantilla
from scipy.signal import correlate, find_peaks, lfilter


mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead =(mat_struct['ecg_lead']).flatten()
N = len(ecg_one_lead)


####################
# tipos de filtro ##
####################


tipo = 'bandpass'

cant_coef = 5 #para que el retardod e grupo sea constante y entero
antisymmetric = False

# plantilla
fs = 1000
nyq_frec = fs/2
ripple = 0.5#dB osea debería modificar la plantilla si uso filtfilt

attenuation =40#dB

    
fs1 = 1.0 # 
fp1 = 0.1 # Hz
fp2 = 50 # 
fs2 = 35.0 # Hz
    
fstop = [fs1,fs2]  #comienzo banda de atenuación, hasta 50 (interferencia de la red eléctrica)
fpass = [fp1,fp2] # banda de paso wp
    

# frecs y gains normalizados a Nyquist
frecs =[0.0, fs1, fp1, fp2, fs2, 1]
gains = [-np.inf, -attenuation, -ripple, -ripple,  -attenuation, -np.inf]

gains = 10**(np.array(gains)/20)
# diseño FIr
ventana = 'blackmanharris'
num = sig.firwin2(cant_coef, frecs, gains, window=ventana, antisymmetric=antisymmetric)
den = 1.0


wrad, hh = sig.freqz(num, den)
ww = wrad / np.pi

plt.figure(1)

plt.plot(ww, 20 * np.log10(abs(hh)), label=tipo+ '-' + ventana)

plot_plantilla(tipo = tipo , fpass = fpass, ripple = ripple , fstop = fstop, attenuation = attenuation, fs = fs)

plt.title('FIR por ventanas')
plt.xlabel('Frequencia normalizada')
plt.ylabel('Modulo [dB]')
plt.grid(True)
plt.show()

plt.figure(2)

phase = np.unwrap(np.angle(hh))

plt.plot(ww, phase, label=tipo+ '-' + ventana)

plt.title('FIR por ventanas')
plt.xlabel('Frequencia normalizada')
plt.ylabel('Fase [rad]')
plt.grid(True)
plt.show()

ecg_filt = sig.filtfilt(num, den, ecg_one_lead)

#segmento
segmento = ecg_one_lead[0:120]

# Calcular correlación cruzada entre señal filtrada y ventana
correlacion = correlate(ecg_filt, segmento, mode='same')


peaks, _ = find_peaks(correlacion, threshold = 35, distance = 300)


plt.figure(figsize=(12,8))

plt.subplot(3,1,1)
plt.plot(ecg_one_lead, label='ECG original')
plt.title('ECG original')
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(ecg_filt, label='ECG filtrado')
plt.title('ECG filtrado con FIR')
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(correlacion, label='Correlación')
plt.plot(peaks, correlacion[peaks], 'rx', label='Picos detectados')
plt.title('Correlación de ECG filtrado con ventana de referencia y picos detectados')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
