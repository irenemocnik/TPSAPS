# -*- coding: utf-8 -*-
"""
Created on Wed May  7 19:53:42 2025

@author: iremo
"""

#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on Wed Nov  8 19:55:30 2023

@author: mariano
"""

import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
   

def vertical_flaten(a):

    return a.reshape(a.shape[0],1)

def butter_lowpass_filter(senal, cutoff, fs, order=4):
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    b, a = sig.butter(order, norm_cutoff, btype='low', analog=False)
    return sig.filtfilt(b, a, senal)
#%%

####################################
# Lectura de pletismografía (PPG)  #
####################################

fs_ppg = 400 # Hz

##################
## PPG con ruido
##################

#Cargar el archivo CSV como un array de NumPy
ppgruido = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1)  # Omitir la cabecera si existe


##################
## PPG sin ruido
##################

ppgsinruido = np.load('ppg_sin_ruido.npy')

ppgsinruido = ppgsinruido.ravel() #aplana un array multidimensional a una sola dimensión.
ppgruido = ppgruido.ravel()

ppgvar = np.var(ppgsinruido) #varianza
#ppgvar = np.var(ppgruido)


N = len(ppgsinruido) 
nperseg = N // 20      #  largo total/100
noverlap = nperseg // 2     # solapo 50%
#el sesgo disminuye a medida que se incrementa la longitud del segmento
#la varianza se reduce cuando se incrementa la cantidad de segmentos
#el solapamiento reduce la independencia entre los segmentos.

#la eleccion de ventana trae un tradeoff entre el ancho del lobulo principaly la altura de los lobulos laterales.
#mayor ancho del lobulo = lobulos laterales bajos, = poca varianza y baja resolucion.

#los lobulos laterales determinan el rango dinamico, el ratio entre la amplitud mayor y la menor que podemos distinguir.

f_ppg, Pxxsinruido = sig.welch(ppgsinruido, fs=fs_ppg, window='hamming', nperseg=nperseg, noverlap=noverlap, detrend='linear', scaling='density')

Pxxsinruido_normalizado = Pxxsinruido / np.var(ppgsinruido)

#hamming:reduce lóbulos laterales, mejora contraste).
#normalizamos dividiendo por la varianza, esto impone una escala estandar.


area_acum_sinruido = np.cumsum(Pxxsinruido_normalizado)
bin95_sinruido = np.where(area_acum_sinruido >= 0.95 * area_acum_sinruido)[0][0]
f95_sinruido = f_ppg[bin95_sinruido]
print("Frecuencia 95% (sin ruido):", f95_sinruido)


#%%Corroboramos teo de parseval

energiaTiempo = np.sum(ppgsinruido**2)
X = np.fft.fft(ppgsinruido)
energiaFreq = np.sum(np.abs(X)**2) / N

print("EN Tiempo =", energiaTiempo)
print("EN FRECUENCIA =", energiaFreq)
print("RESTA =", energiaTiempo - energiaFreq)

#%% Señal ruidosa

f_ruido, Pxx_ruido = sig.welch(ppgruido, fs=fs_ppg, window='hamming', nperseg=nperseg, noverlap=noverlap, detrend='linear', scaling='density')
Pxx_ruido_norm = Pxx_ruido / np.var(ppgruido)
#uso el valor del bin95 en freq para cortar el filtro pasabajos
cutoff = f95_sinruido  # corte definido por la señal sin ruido



area_acum_ruido = np.cumsum(Pxx_ruido_norm)
bin95_ruido = np.where(area_acum_ruido >= 0.95*area_acum_ruido)[0][0]
f95_ruido = f_ruido[bin95_ruido]
print("Frecuencia 95% (con ruido):", f95_ruido)

ppg_filtrada = butter_lowpass_filter(ppgruido, cutoff, fs_ppg)

f_filt, Pxx_filt = sig.welch(ppg_filtrada, fs=fs_ppg, window='hamming',
                             nperseg=nperseg, noverlap=noverlap, detrend='linear',
                             scaling='density')
Pxx_filt_norm = Pxx_filt / np.var(ppg_filtrada)


area_acum_filt = np.cumsum(Pxx_filt_norm) 
bin95_filt = np.where(area_acum_filt >= 0.95*area_acum_filt)[0][0]
f95_filt = f_filt[bin95_filt]      
print("Frecuencia 95% (filtrada):", f95_filt)

#%%

t = np.arange(len(ppgruido)) / fs_ppg
plt.figure()
plt.plot(t, ppgruido, label='Original con ruido', alpha=0.5)
plt.plot(t, ppg_filtrada, label='Filtrada (Butterworth)', linewidth=2)
plt.title('Señal PPG: original vs filtrada')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.legend()
plt.grid()



 #%% señal original despues de aplicarle un filtro.
plt.figure()
plt.plot(f_ppg, 10 * np.log10(Pxxsinruido_normalizado), label='Sin ruido')
plt.plot(f_ruido, 10 * np.log10(Pxx_ruido_norm), label='Con ruido')
plt.plot(f_filt, 10 * np.log10(Pxx_filt_norm), label='Filtrada')
plt.title('PSD comparadas')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad espectral [dB]')
plt.legend()
plt.grid()

