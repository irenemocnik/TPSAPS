# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 19:22:17 2025

@author: iremo
"""

#%% módulos y funciones a importar

import numpy as np
from scipy import signal
from scipy.fft import fft, fftshift
import matplotlib.pyplot as plt




def seno(Vmax, dc, ff, ph, N, fs):
    Ts = 1/fs #Paso de tiempo
    Tiempoint = N * Ts # Inervalo total de tiempo
    tt = np.arange(start = 0, stop = Tiempoint, step = Ts)
    xx = Vmax * np.sin( 2 * np.pi * ff * tt ) + dc
    return tt, xx

Vmax=1/(np.sqrt(2))
dc=0
ff=250.5
ph=0
fs = 1000 # frecuencia de muestreo (Hz)
N = 1000 # cantidad de muestras
df=fs/N

Ts = 1/fs #Paso de tiempo
Tiempoint = N * Ts # Inervalo total de tiempo 

tt, xx = seno(Vmax, dc, ff, ph, N, fs)

# np.random.normal
# np.random.uniform


#%% Datos de la simulación
# Datos del ADC
B =  8# bits
Vf = 2 # rango simétrico de +/- Vf Volts
q =  Vf/(2**7) # paso de cuantización de q Volts

#BW=40000
# # datos del ruido (potencia de la señal normalizada, es decir 1 W)

pot_ruido_cuant = 1 #(50^2)*(10^(-18))*BW Eso sería del conversor que vimos # Watts 
kn = 1 # escala de la potencia de ruido analógico
pot_ruido_analog = pot_ruido_cuant * kn # 



#%% Experimento: 
"""
   Se desea simular el efecto de la cuantización sobre una señal senoidal de 
   frecuencia 1 Hz. La señal "analógica" podría tener añadida una cantidad de 
   ruido gausiano e incorrelado.
   
   Se pide analizar el efecto del muestreo y cuantización sobre la señal 
   analógica. Para ello se proponen una serie de gráficas que tendrá que ayudar
   a construir para luego analizar los resultados.
   
"""

# np.random.normal
# np.random.uniform


# Señales

analog_sig = xx # señal analógica sin ruido
nn = np.random.normal(0,np.sqrt((q**2)/12), N)
sr = analog_sig + nn # señal analógica de entrada al ADC (con ruido analógico)
srq = q*np.round(sr/q)# señal cuantizada
ventana = signal.windows.flattop(N)
xxventana = srq*ventana
desvio=np.std(ventana)
xxventananormalizada=xxventana/desvio


nq = srq - sr # señal de ruido de cuantización
varianzan= np.var(nn)
print(varianzan)



#%% Visualización de resultados

# cierro ventanas anteriores
#plt.close('all')

##################
# Señal temporal
##################

plt.figure(1)

plt.plot(tt, srq, lw=2, linestyle='', color='blue', marker='o', markersize=5, markerfacecolor='blue', markeredgecolor='blue', fillstyle='none', label='ADC out (diezmada)')
plt.plot(tt, sr, lw=1, color='black', marker='x', ls='dotted', label='$ s $ (analog)')

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
ft_XVENTANA = 1/N*np.fft.fft( xxventananormalizada)

# grilla de sampleo frecuencial
ff = np.linspace(0, (N-1)*df, N)

bfrec = ff <= fs/2 #hasta nyquist, por la simetría de la fft, lo demás es redundante.

Nnq_mean = np.mean(np.abs(ft_Nq)**2)
nNn_mean = np.mean(np.abs(ft_Nn)**2)

# plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_As[bfrec])**2), color='yellow', ls='dotted', label='$ s $ (sig.)' )
# plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([nNn_mean, nNn_mean]) ), '--r', label= '$ \overline{n} = $' + '{:3.1f} dB'.format(10* np.log10(2* nNn_mean)) )
# plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_SR[bfrec])**2), ':g', label='$ s_R = s + n $' )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Srq[bfrec])**2), lw=2, label='$ s_Q = Q_{B,V_F}\{s_R\}$' )
# plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([Nnq_mean, Nnq_mean]) ), '--c', label='$ \overline{n_Q} = $' + '{:3.1f} dB'.format(10* np.log10(2* Nnq_mean)) )
# plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nn[bfrec])**2), ':r')
# plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nq[bfrec])**2), ':c')
# plt.plot( np.array([ ff[bfrec][-1], ff[bfrec][-1] ]), plt.ylim(), ':k', label='BW', lw = 0.5  )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_XVENTANA[bfrec])**2),  color='orange', ls='dotted', label='$ x ventana normalizada $ (sig.)' )


plt.title('Ventaneo'.format(B, Vf, q) )
plt.ylabel('Densidad de Potencia [dB]')
plt.xlabel('Frecuencia [Hz]')
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