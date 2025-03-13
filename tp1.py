# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 23:40:38 2025

@author: iremo
"""

#%%
import numpy as np
import matplotlib.pyplot as plt



fs = 1000 # Hz
N = fs

pasof = fs / N # Hz

Ts = 1/fs #Paso de tiempo
Tiempoint = N * Ts # Inervalo total de tiempo


ff = 1 # Hz
Vmax = 1
ph =0
nn = N
dc = 0

def mi_funcion_sen(Vmax, dc, ff, ph, nn, fs):
    tt = np.arange(start = 0, stop = Tiempoint, step = Ts)
    xx = Vmax * np.sin( 2 * np.pi * ff * tt ) + dc
    return tt, xx

tt, xx = mi_funcion_sen(Vmax, dc, ff, ph, nn, fs)
plt.plot(tt, xx, label="1 Hz")
#%%
#otras frecuencias
ff2=500
ff3=999
ff4=1001
ff5=2001
tt, xx = mi_funcion_sen(Vmax, dc, ff2, ph, nn, fs)
plt.plot(tt, xx, label="500 Hz")

tt, xx = mi_funcion_sen(Vmax, dc, ff3, ph, nn, fs)
plt.plot(tt, xx, label="999 Hz")

tt, xx = mi_funcion_sen(Vmax, dc, ff4, ph, nn, fs)
plt.plot(tt, xx, label="1001 Hz")

tt, xx = mi_funcion_sen(Vmax, dc, ff5, ph, nn, fs)
plt.plot(tt, xx, label="2001 Hz")


plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.grid()
plt.legend()
plt.show()


"""
Para una frecuencia de muestreo de 1000Hz:
    500 Hz es la frecuencia de Nyquist= 1000/2, por lo que
    cualquier señal con ff>500Hz no se representa correctamente.
    Los gráficos para f={1,1001,2001} son iguales -> las señales de frecuencia:
        f=fs+x se ven como si fueran de frecuencia f=x.
        (Por este motivo los gráficos que corresponden a ff= 1, 1001 y 2001 Hz están superpuestos)
    De manera análoga, las señales con frecuencia:
        f=fs-x, se ven como una señal de f=x invertida
        (Po este motivo, la señal de ff=999 se ve como la señal de ff=1Hz pero espejada/invertida)
    Por último, la señal de ff=500Hz=frecuencia de Nyquist se ve plana, esto se debe a que la señal tiene solo 2 puntos por ciclo y se "cancelan"

"""
#%%
#Otra señal
x_cuadrada = np.sign(xx)
plt.plot(tt, x_cuadrada, label="Señal Cuadrada")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.grid()
plt.legend()
plt.show()

"""
La señal cuadrada usa los últimos valores en el vector xx (ff=ff5=2001 Hz)
Debido al aliasing, la representación gráfica equivale a una señal de 1Hz.
"""
