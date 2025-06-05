# -*- coding: utf-8 -*-
"""
Created on Wed May  7 19:53:42 2025

@author: iremo
"""


import numpy as np
from scipy import signal as sig
import scipy.io as sio

import matplotlib.pyplot as plt
   

def vertical_flaten(a):

    return a.reshape(a.shape[0],1)





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
df = fs_ppg/N
ff = np.linspace(0, (N-1)*df, N)
bfrec = ff <= fs_ppg/2

#el sesgo disminuye a medida que se incrementa la longitud del segmento
#la varianza se reduce cuando se incrementa la cantidad de segmentos
#el solapamiento reduce la independencia entre los segmentos.

#la eleccion de ventana trae un tradeoff entre el ancho del lobulo principaly la altura de los lobulos laterales.
#mayor ancho del lobulo = lobulos laterales bajos, = poca varianza y baja resolucion.

#los lobulos laterales determinan el rango dinamico, el ratio entre la amplitud mayor y la menor que podemos distinguir.

f_ppg, Pxxsinruido = sig.welch(ppgsinruido, fs=fs_ppg, window='hamming', nperseg=nperseg, noverlap=noverlap, detrend='linear', scaling='density')


ventana=sig.windows.hamming(N)
Xventana = ppgsinruido * ventana
Xhamming = np.fft.fft(Xventana)
periodograma = np.abs(Xhamming)**2 / N


area_total = np.sum(Pxxsinruido) 

Pxxsinruido_normalizado = Pxxsinruido / area_total 

suma= np.sum(Pxxsinruido_normalizado) #Verifica q este correctamente normalizado


#hamming:reduce lóbulos laterales, mejora contraste).
#normalizamos dividiendo por la varianza, esto impone una escala estandar.



# quiero obtener el bin donde se acumula el 95% de la energia

area = np.cumsum(Pxxsinruido_normalizado)
bin95 = np.where(area >= 0.95)[0][0] #busco el primer bin donde la energia acumulada supera el 95%
freq95 = f_ppg[bin95] 

print(f"Frecuencia donde se acumula el 95% de la energía: {freq95:.1f} Hz")




#%%Corroboramos teo de parseval

energiaTiempo = np.sum(ppgsinruido**2)
X = np.fft.fft(ppgsinruido)
energiaFreq = np.sum(np.abs(X)**2) / N

print("EN Tiempo =", energiaTiempo)
print("EN FRECUENCIA =", energiaFreq)
print("RESTA =", energiaTiempo - energiaFreq)


plt.figure()
plt.plot(f_ppg, 10 * np.log10(Pxxsinruido_normalizado), label='Welch', color='blue')
plt.plot(ff[bfrec],  10 * np.log10(periodograma[bfrec]), label='Periodograma', color='red')

plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad espectral de potencia [V^2/Hz]')
plt.title('PPG sin ruido')
plt.legend()


##################
## ECG sin ruido
##################

fs_ecg = 300

ecg_one_lead = np.load('ecg_sin_ruido.npy')
ecg_one_lead = ecg_one_lead.ravel()

N_ecg = len(ecg_one_lead)
nperseg_ecg = N_ecg // 20
noverlap_ecg = nperseg_ecg // 2

f_ecg, Pxx_ecg = sig.welch(ecg_one_lead, fs=fs_ecg, window='hamming',
                           nperseg=nperseg_ecg, noverlap=noverlap_ecg,
                           detrend='linear', scaling='density')

mitadpsd=Pxx_ecg[:N//2].ravel()
ffmitad=f_ecg[:N//2]

energia_total = np.sum(mitadpsd)
energia_acumulada = np.cumsum(mitadpsd) / energia_total
indice_95 = np.where(energia_acumulada >= 0.95)[0][0]
freq95_ecg = ffmitad[indice_95]
indice_98 = np.where(energia_acumulada >= 0.98)[0][0]
freq98_ecg = ffmitad[indice_98]




# area_ecg = np.cumsum(Pxx_ecg) / area_total_ecg
# bin95_ecg = np.where(area_ecg >= 0.95)[0][0]
# freq95_ecg = f_ecg[bin95_ecg]

print(f"[ECG] Frecuencia donde se acumula el 95% de la energía: {freq95_ecg:.1f} Hz")


plt.figure()
plt.plot(f_ecg, 10 * np.log10(Pxx_ecg), label='Welch', color='blue')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad espectral de potencia [V^2/Hz]')
plt.title("PSD - ECG")
plt.legend()

#%%

####################
# Lectura de audio #
####################

# Cargar el archivo CSV como un array de NumPy
fs_audio, wav_data = sio.wavfile.read('la cucaracha.wav')
# fs_audio, wav_data = sio.wavfile.read('prueba psd.wav')
# fs_audio, wav_data = sio.wavfile.read('silbido.wav')
wav_data = wav_data.ravel()


N_audio = len(wav_data)
nperseg_audio = N_audio // 20
noverlap_audio = nperseg_audio // 2

f_audio, Pxx_audio = sig.welch(wav_data, fs=fs_audio, window='hamming',
                               nperseg=nperseg_audio, noverlap=noverlap_audio,
                               detrend='linear', scaling='density')

area_total_audio = np.sum(Pxx_audio)
Pxx_audio_normalizado = Pxx_audio / area_total_audio
suma_audio = np.sum(Pxx_audio_normalizado)

area_audio = np.cumsum(Pxx_audio_normalizado)
bin95_audio = np.where(area_audio >= 0.95)[0][0]
freq95_audio = f_audio[bin95_audio]

print(f"[Audio] Frecuencia donde se acumula el 95% de la energía: {freq95_audio:.1f} Hz")

plt.figure()
plt.plot(f_audio, 10 * np.log10(Pxx_audio), label='Welch', color='blue')
plt.title("PSD - Audio")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Densidad espectral (V²/Hz)")
plt.grid()