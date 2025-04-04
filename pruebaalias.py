# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 20:24:35 2025

@author: iremo
"""

import numpy as np
import matplotlib.pyplot as plt

# Función para generar la señal seno
def seno(vmax, dc, ff, ph, N, fs):
    ts = 1/fs  # Paso de tiempo
    tiempoint = N * ts  # Intervalo total de tiempo
    tt = np.arange(start=0, stop=tiempoint, step=ts)
    xx = vmax * np.sin(2 * np.pi * ff * tt + ph) + dc
    return tt, xx

# Parámetros
vmax = np.sqrt(2)
fs = 1000  # Frecuencia de muestreo (Hz)
N = 1000   # Cantidad de muestras
ts = 1/fs  # Tiempo de muestreo
df = fs/N  # Resolución espectral
dc = 0
ff_original = 50  # Frecuencia dentro del rango de Nyquist
ff_alias = 700    # Frecuencia fuera del rango de Nyquist

# Cálculo de aliasing según el teorema de Nyquist
ff_aliasado = np.abs(ff_alias - np.round(ff_alias / fs) * fs)

# Generación de señales
tt, xx_original = seno(vmax, dc, ff_original, 0, N, fs)
tt, xx_alias = seno(vmax, dc, ff_alias, 0, N, fs)
tt, xx_aliasado = seno(vmax, dc, ff_aliasado, 0, N, fs)

# Visualización en el dominio del tiempo
plt.figure(figsize=(10, 5))
plt.plot(tt, xx_original, label=f'Señal Original {ff_original} Hz', color='blue')
plt.plot(tt, xx_alias, label=f'Señal Muestreada {ff_alias} Hz', color='red', linestyle='dotted')
plt.plot(tt, xx_aliasado, label=f'Aliasing a {ff_aliasado} Hz', color='green', linestyle='dashed')
plt.title("Aliasing en el Dominio del Tiempo")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.legend()
plt.xlim(0, 0.05)
plt.grid()
plt.show()

# Transformada de Fourier
ft_original = np.abs(np.fft.fft(xx_original)/N)**2
ft_alias = np.abs(np.fft.fft(xx_alias)/N)**2
ft_aliasado = np.abs(np.fft.fft(xx_aliasado)/N)**2

# Eje de frecuencias
frecuencias = np.fft.fftfreq(N, d=ts)
bfrec = frecuencias >= 0  # Tomamos solo la mitad positiva

# Visualización en el dominio de la frecuencia
plt.figure(figsize=(10, 5))
plt.plot(frecuencias[bfrec], 10*np.log10(ft_original[bfrec]), label=f"Frecuencia Real {ff_original} Hz", color='blue')
plt.plot(frecuencias[bfrec], 10*np.log10(ft_alias[bfrec]), label=f"Frecuencia Muestreada {ff_alias} Hz", color='red', linestyle='dotted')
plt.plot(frecuencias[bfrec], 10*np.log10(ft_aliasado[bfrec]), label=f"Alias en {ff_aliasado} Hz", color='green', linestyle='dashed')
plt.axvline(fs/2, color='black', linestyle='dashdot', label='Frecuencia de Nyquist')
plt.title("Aliasing en el Dominio de la Frecuencia")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad de Potencia [dB]")
plt.legend()
plt.grid()