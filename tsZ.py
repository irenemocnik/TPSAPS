# -*- coding: utf-8 -*-
"""
Created on Fri May 30 12:43:20 2025

@author: iremo
"""


import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt

tznum = [1, 1, 1, 1] #  1 + z^{-1} + z^{-2} + z^{-3}
tzden = [1, 0, 0, 0]

w, h = sig.freqz(tznum, tzden)
freqN = w / np.pi

    
plt.figure (1)
plt.plot(freqN, 20 * np.log10(np.abs(h)))
plt.xlabel('Frequencia normalizada')
plt.ylabel('Modulo [dB]')
plt.title('Respuesta de módulo en frecuencia')
plt.grid()

plt.figure (2)
plt.plot(freqN, np.angle(h))
plt.xlabel('Frequencia normalizada')
plt.ylabel('Fase [rad]')
plt.title('Respuesta de fase en frecuencia')
plt.grid()

plt.figure (3)
retardoGrupo = -np.diff(np.unwrap(np.angle(h))) / np.diff(w)
plt.plot(freqN[1:], retardoGrupo)
plt.title('Retardo de grupo')
plt.xlabel('Frequencia normalizada')
plt.ylabel('Retardo de grupo')
plt.grid()
    
# Find maximum delay and set plot limits
#max_delay_samples = np.ceil(np.max(gd))
#ax3.set_ylim(0, max_delay_samples)

# polos y ceros
# Diagrama de polos y ceros
tfz = sig.TransferFunction(tznum, tzden, dt=1)  # Sistema discreto

plt.figure(4)
plt.plot(np.real(tfz.zeros), np.imag(tfz.zeros), 'o', color='blue', markersize=10,
         fillstyle='none', markeredgewidth=2, label='Ceros')
plt.plot(np.real(tfz.poles), np.imag(tfz.poles), 'x', color='red', markersize=10,
         markeredgewidth=2, label='Polos')

circulo = plt.Circle((0, 0), 1, color='gray', linestyle='--', fill=False)
plt.gca().add_patch(circulo)
plt.axis('equal')
plt.title('Polos y ceros')
plt.xlabel('Re')
plt.ylabel('Im')
plt.legend()
plt.grid()

plt.show()