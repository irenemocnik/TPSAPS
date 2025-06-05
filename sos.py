# -*- coding: utf-8 -*-
"""
Created on Wed May 28 18:15:12 2025

@author: iremo
"""

import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
   
import scipy.io as sio
from scipy.io.wavfile import write
from pytc2.sistemas_lineales import plot_plantilla

#%%diseño filtro

aprox_name = 'butter'
# aprox_name = 'cheby1'
# aprox_name = 'cheby2'
# aprox_name = 'ellip'
fs = 1000
nyq_frec = fs/2
fpass= np.array([1.0,35.0]) #banda de paso wp
ripple = 1#dB osea debería modificar la plantilla si uso filtfilt
fstop= ([.1,50.]) #comienzo banda de atenuación, hasta 50 (interferencia de la red eléctrica)
attenuation =40#dB

#para una comparacio njusta, cambio los parametros para el filt filt
# s = 1000
# nyq_frec = fs/2
# fpass= np.array([1.0,35.0]) #banda de paso wp
# ripple = 0.5#dB osea debería modificar la plantilla si uso filtfilt
# fstop= ([.1,50.]) #comienzo banda de atenuación, hasta 50 (interferencia de la red eléctrica)
# attenuation =20#dB



#esperamos una rta lineal
#esperamos retardo cte, porque introduce un retardo de fase
#sabemos por fourier que si pasamos por un sistema LTI, si tenemos una demora cte, se introduce un producto por una exponencial, entonces se suma una fase cte


mi_sos = sig.iirdesign(fpass, fstop, ripple, attenuation, ftype=aprox_name, output='sos',fs=fs)
# las columnas 3,4y5 son los coeficiente a0,a1,a2 y de la 0 a la 2 son los coef b

#%%plantilla de diseño, para analizarlo
npoints = 1000 #asi evalua equiespaciado

#para obtner mayor resuloción antesd e la bandad de paso, necesito un muestreo log => a freqz le puedo pasar un vector.
w_rad = np.append(np.logspace(-2,0.8,250), np.logspace(0.9,1.6,250))
w_rad = np.append(w_rad, np.linspace(40, nyq_frec, 500, endpoint=True) )/nyq_frec * np.pi

w, hh = sig.sosfreqz(mi_sos, worN=w_rad) #worN le puedo pasar un entero o un vector 
##hh pertenece a complejos, w son los valores en los que calcula el vector de complejos
fase = np.angle(hh)
modulo = np.abs(hh)
group = -np.diff(fase) / np.diff(w)



plt.figure()
plt.plot(w/np.pi*nyq_frec, np.angle(hh), label='mi_sos')
plt.title('Plantilla de diseño')
plt.xlabel('Frecuencia normalizada a Nyq [#]')
plt.ylabel('Amplitud [dB]')

plt.figure()
plt.plot(w/np.pi*nyq_frec, 20*np.log10(np.abs(hh)+1e-15), label='mi_sos')
plt.title('Plantilla de diseño')
plt.xlabel('Frecuencia normalizada a Nyq [#]')
plt.ylabel('Amplitud [dB]')
# filtro anterior, como referencia
# w, mag, _ = my_digital_filter.bode(npoints)
# plt.plot(w/w_nyq, mag, label=my_digital_filter_desc)

plt.figure()
plt.plot(w/np.pi*nyq_frec, 20*np.log10(np.abs(hh)+1e-15), label='mi_sos')
gd = -np.diff(np.unwrap(np.angle(hh))) / np.diff(w)
plt.plot(w[1:]/np.pi*nyq_frec, gd)
plt.title('Retardo de grupo')
plt.xlabel('Frecuencia normalizada a Nyq [#]')
plt.ylabel('Retardo de grupo [muestras]')



#%%señal ECG
mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead =(mat_struct['ecg_lead']).flatten()
N = len(ecg_one_lead)

#filtro ECG, con funcion sosfiltfilt
ECG_filtfilt= sig.sosfiltfilt(mi_sos, ecg_one_lead)

plt.figure()
plt.plot(ecg_one_lead,label='ECG original')
plt.plot(ECG_filtfilt, label='ECG filtrado')
plt.title('ECG antes y después del filtrado')
plt.legend()
plt.show()

#si estuvieramos con una señal real que sale de un adc, usariamos un filtro NO bidireccional
#ademas de la rta en frecuecnia, nos interesa la rta temporal
#poenmos a prueba como funciona en la banda de paso y la banda de rechazo
#buscamos una region dentro de la banda de pas: buscamos una region limpia dentro de la banda de paso. 
#para probar la banda derechazo superior, vemos señales de alta frec en el regimen temporal (con mucha interferencia)
#para probar la badnda de rechazo inferior, buscamos fluctuaciones lentas

#ECG_filt = sig.sosfilt(mi_sos, ecg_one_lead) 
ECG_filt = sig.sosfiltfilt(mi_sos, ecg_one_lead)  #neutraliza demora y dist de fase
#con freqz podemos agregar que en vez de devolver abs devuelva ANGLE, eso

#si sigo con filt, puedo mirar a ojo la diferencia entre dos picos (parece aprox 68) y establezco esta demora como una cte, es aproximado pero es suficiente

#anula distorsion de fase, anula la demora
demora = 0 
#demora = 68

#se generan oscilaciones por la forma de la rta al impulso del filtro
#se podria aislar este efecto usanod una respuesta menos abrupt

fig_dpi = 150
cant_muestras =len(ecg_one_lead)
ECG_f_win = ECG_filt
fig_sz_x = 10
fig_sz_y = 10

regs_interes = ( 
        np.array([5, 5.2]) *60*fs, # minutos a muestras
        np.array([12, 12.4]) *60*fs, # minutos a muestras
        np.array([15, 15.2]) *60*fs, # minutos a muestras
        )

for ii in regs_interes:
    
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
    
    plt.figure(figsize=(fig_sz_x, fig_sz_y), dpi= fig_dpi, facecolor='w', edgecolor='k')
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    #plt.plot(zoom_region, ECG_f_butt[zoom_region], label='Butter')
    plt.plot(zoom_region, ECG_f_win[zoom_region + demora], label='Win')
    
    plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
    
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
            
    plt.show()
regs_interes = ( 
        [4000, 5500], # muestras
        [10e3, 11e3], # muestras
        )

for ii in regs_interes:
    
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
    
    plt.figure(figsize=(fig_sz_x, fig_sz_y), dpi= fig_dpi, facecolor='w', edgecolor='k')
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    #plt.plot(zoom_region, ECG_f_butt[zoom_region], label='Butter')
    plt.plot(zoom_region, ECG_f_win[zoom_region + demora], label='Win')
    
    plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
    
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
            
    plt.show()
    
    
    
    
    
    #esperamos una rta lineal
    #esperamos retardo cte, porque introduce un retardo de fase
    #sabemos por fourier que si pasamos por un sistema LTI, si tenemos una demora cte, se introduce un producto por una exponencial, entonces se suma una fase cte

#para un h iir, la rta de retarno no es cte, tenemos demora diferente para c ancho de banda, entonces lagunas frec cambian su fase relativa cn respecto de otras.
