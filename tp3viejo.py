# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 15:29:01 2025

@author: iremo
"""

#%% módulos y funciones a importar

import numpy as np
import matplotlib.pyplot as plt




def seno(Vmax, dc, ff, ph, N, fs):
    Ts = 1/fs #Paso de tiempo
    Tiempoint = N * Ts # Inervalo total de tiempo
    tt = np.arange(start = 0, stop = Tiempoint, step = Ts)
    xx = Vmax * np.sin( 2 * np.pi * ff * tt ) + dc
    return tt, xx

Vmax=np.sqrt(2)
dc=0
ff=1
ph=0
fs = 1000 # frecuencia de muestreo (Hz)
N = 1000 # cantidad de muestras
df=fs/N

Ts = 1/fs #Paso de tiempo
Tiempoint = N * Ts # Inervalo total de tiempo 

tt, xx = seno(Vmax, dc, ff, ph, N, fs)

#var=np.var(xx) #chequear que esta correctamente normalizada


#%% Datos de la simulación
# Datos del ADC
B =  4# bits
Vf = 2 # rango simétrico de +/- Vf Volts
q =  Vf/(2**(B-1))# paso de cuantización de q Volts

print(q)

Pq = (q**2)/12 
kn = 1 # escala de la potencia de ruido analógico
Pn = Pq * kn #potencia para la señal de ruido normal



#%% Experimento: 

analog_sig = xx # señal analógica sin ruido

nn = np.random.normal(0,np.sqrt(Pn), N)

sr = analog_sig + nn # señal analógica de entrada al ADC (con ruido analógico)

srq = q*np.round(sr/q)# señal cuantizada

nq = srq - sr # señal de ruido de cuantización


#%% Visualización de resultados

# cierro ventanas anteriores
#plt.close('all')

##################
# Señal temporal
##################

plt.figure(1)

plt.plot(tt, srq, lw=1, color='blue', marker='.', markersize=2, markerfacecolor='blue', markeredgecolor='blue', fillstyle='none', label='Srq=Salida ADC')
plt.plot(tt, sr, lw=1, color='green', markersize=2, marker='.', label='$ sr = s + nn $ (Señal analógica)')
plt.plot(tt, xx, lw=1, color='black', ls='dotted', label='$ s $ (Señal pura)')

plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
axes_hdl = plt.gca()
axes_hdl.legend()
plt.show()


###########
# Espectro
###########

plt.figure(2)
ft_SR = 1/N*np.fft.fft( sr)
ft_Srq = 1/N*np.fft.fft( srq)
ft_As = 1/N*np.fft.fft( analog_sig)
ft_Nq = 1/N*np.fft.fft( nq)
ft_Nn = 1/N*np.fft.fft( nn)

# grilla de sampleo frecuencial
ff = np.linspace(0, (N-1)*df, N)

bfrec = ff <= fs/2 #hasta nyquist, por la simetría de la fft, lo demás es redundante.

Nnq_mean = np.mean(np.abs(ft_Nq)**2)
nNn_mean = np.mean(np.abs(ft_Nn)**2)

plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_As[bfrec])**2), color='orange', ls='dotted', label='$ s $ (analog)' )
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([nNn_mean, nNn_mean]) ),  color='red', ls='dotted', label= '$ \overline{n} = $' + '{:3.1f} dB'.format(10* np.log10(2* nNn_mean)) )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_SR[bfrec])**2), ':g', color='green', ls = 'dotted', label='$ s_R = s + n $' )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Srq[bfrec])**2), color='blue',lw=1, label='$ s_Q = Q_{B,V_F}\{s_R\}$' )
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([Nnq_mean, Nnq_mean]) ), '--c', color='cyan', label='$ \overline{n_Q} = $' + '{:3.1f} dB'.format(10* np.log10(2* Nnq_mean)) )
# plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nn[bfrec])**2), ':r')
# plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nq[bfrec])**2), ':c')
# plt.plot( np.array([ ff[bfrec][-1], ff[bfrec][-1] ]), plt.ylim(), ':k', label='BW', lw = 0.5  )

plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
plt.ylabel('Densidad de Potencia [dB]')
plt.xlabel('Frecuencia [Hz]')
plt.ylim([-80, 0]) 

axes_hdl = plt.gca()
axes_hdl.legend()

#############
# Histograma
#############

plt.figure(3)
bins = 10
plt.hist(nq.flatten()/(q), bins=bins)
plt.plot( np.array([-1/2, -1/2, 1/2, 1/2]), np.array([0, N/bins, N/bins, 0]), '--r' )
plt.title( 'Ruido de cuantización para {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q))

plt.xlabel('Pasos de cuantización (q) [V]')