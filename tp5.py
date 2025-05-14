# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 19:04:12 2025

@author: iremo
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

#%% Función para generar la señal seno
def seno(N, R, SNR):
    fs=1000
    df= fs/N
    w0 = fs/4
    a1=np.sqrt(2)

    t = np.arange(0,1,1/N).reshape(N, 1)
    tt=np.tile(t, (1,R))

    fr = np.random.uniform(-1/2, 1/2, size=(1, R))
    w1 = w0 + fr * df

    S = a1*np.sin(2*np.pi*w1*tt)
    potseñal = np.mean(S**2)
    potruido = potseñal / (10**(SNR/10))

    ruido=np.random.normal(0, np.sqrt(potruido), size=S.shape)
    Xr = S + ruido

    return Xr, w1

N = 1000
R = 200
SNR = 20
fs = 1000
df = fs / N
Xr, w1 = seno(N=N, R=R, SNR=SNR)

ff = np.linspace(0, (N-1)*df, N)
bfrec = ff <= fs/2
w0 = fs/4
a1= np.sqrt(2)

   

# tal ue potencia acumualada/ potencia total sea mayor o igual a 98%



nperseg = N//4
noverlap = N//8
f, Pxx = sig.welch(Xr, fs=1000, window='hann', nperseg = nperseg, noverlap = None, nfft = 4*N, detrend = 'constant', return_onesided=True,scaling = 'density', axis=-1, average= 'mean')

Xr_fft = np.fft.fft(Xr, axis=0)
MAG_Xr_fft = 1/N * np.abs(Xr_fft)


Npad = 4000
dfpad = fs / Npad
ffpad = np.linspace(0, fs - dfpad, Npad)

# Zero-padding de señales
Xrpad = np.zeros((Npad, R))

Xrpad[:N, :] = Xr

# FFT padded
MAG_RECT_pad = 1/N * np.abs(np.fft.fft(Xrpad, axis=0))



aest_RECT = np.zeros(R)
aest_WELCH = np.zeros(R)

for i in range(R):
    k_real = int(np.round(w1[0, i]))  # bin más cercano a la frecuencia real
    aest_RECT[i] = MAG_RECT_pad[k_real, i]
    aest_WELCH[i] = Pxx[k_real, i]
  

#%% Calculo de sesgo y varianza (magnitud)
sesgo_RECT = np.mean(aest_RECT) - a1
sesgo_WELCH = np.mean(aest_WELCH)-a1
var_RECT = np.var(aest_RECT)
var_WELCH = np.var(aest_WELCH) 

#%% Histograma de magnitud
plt.figure(figsize=(10,5))
bins = 30
plt.hist(aest_RECT, bins=bins, alpha=0.5, label=f"Rectangular\nSesgo: {sesgo_RECT:.4f}\nVar: {var_RECT:.4f}")
plt.hist(aest_WELCH, bins=bins, alpha=0.5, label = f"Welch\nSesgo: {sesgo_WELCH:.4f}\nVar: {var_WELCH:.4f}")
plt.xlabel("Valor del estimador")
plt.ylabel("Frecuencia")
plt.title("Comparación de histogramas de estimadores de magnitud en ω₀")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#%% FFT (una realización)
plt.figure(figsize=(10, 5))
plt.plot(ffpad, 10 * np.log10(2 * (MAG_RECT_pad[:N//2, 0])**2), label="Rectangular con padding (N=4000)")
plt.plot(f,10 * np.log10(Pxx[:,0]),label ="Welch")
plt.title("FFT de la 1era realización")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud (dB)")
plt.grid(True)
plt.legend()






