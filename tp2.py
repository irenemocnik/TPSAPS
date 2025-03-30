# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 23:03:53 2025

@author: iremo
"""

import numpy as np
import matplotlib.pyplot as plt

Q = 1
w0= 1
# se analiza un intervalo dos décadas antes y después de la frecuencia de corte. (w/100 hasta w0*100)
w= np.logspace(np.log10(w0/100), np.log10(100*w0), 10000)

def funciona(Q, w0):
    w = np.logspace(np.log10(w0/100), np.log10(100*w0), 10000)
    Ta = (w / (w0 * Q)) / np.sqrt((w / w0)**2 * Q**2 + (1 - (w / w0)**2)**2)
    return w, Ta
def fasea(Q,w0):
    w = np.logspace(np.log10(w0/100), np.log10(100*w0), 10000)
    fa = np.arctan((1 - (w / w0)**2) / ((w / w0) * Q))
    return w, fa

def funcionb(Q, w0):
    w = np.logspace(np.log10(w0/100), np.log10(100*w0), 10000)
    Tb = (1 / (Q * w0)) * (w / np.sqrt((w * (1 + 1 / (Q * w0)))**2 + (w0 / Q)**2))
    return w, Tb
def faseb(Q,w0):
    w = np.logspace(np.log10(w0/100), np.log10(100*w0), 10000)
    fb = np.arctan(w0 * (1 + 1 / (Q * w0)) / w* Q) 
    return w, fb


w, Ta =funciona(Q, w0)
w, fa =fasea(Q,w0)
w, Tb =funcionb(Q, w0)
w, fb =faseb(Q,w0)


plt.figure(1)
plt.semilogx(w, 20*np.log10(Ta), label="Magnitud")
plt.xlabel("Frecuencia (rad/s)")
plt.ylabel("Ganancia (dB)")
plt.title("Diagrama de Bode - Magnitud A")
plt.grid()

plt.figure(2)
plt.semilogx(w, fa)  
plt.ylabel("Fase (radianes)")
plt.xlabel("Frecuencia (rad/s)")
plt.title("Diagrama de Bode - Fase A")
plt.grid()

plt.figure(3)
plt.semilogx(w, 20*np.log10(Tb), label="Magnitud")
plt.xlabel("Frecuencia (rad/s)")
plt.ylabel("Ganancia (dB)")
plt.title("Diagrama de Bode - Magnitud B")
plt.grid()

plt.figure(4)
plt.semilogx(w, fb)  
plt.ylabel("Fase (radianes)")
plt.xlabel("Frecuencia (rad/s)")
plt.title("Diagrama de Bode - Fase B")
plt.grid()

plt.show()




