"""
Created on Tue Jun 10 17:47:28 2025

@author: iremo
"""

import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
   
import scipy.io as sio
from scipy.io.wavfile import write
from pytc2.sistemas_lineales import plot_plantilla
from scipy.signal import sosfiltfilt, filtfilt

def vertical_flaten(a):

    return a.reshape(a.shape[0],1)

#%%


fs_ecg = 1000 # Hz


# ECG con ruido


# para listar las variables que hay en el archivo
sio.whosmat('ecg.mat')
mat_struct = sio.loadmat('ecg.mat')

ecg_one_lead = vertical_flaten(mat_struct['ecg_lead'])
N = len(ecg_one_lead)

 
nyq_frec = fs_ecg/2
ripple = 1 # dB
atenuacion = 40 # dB
 
ws1 = 0.1 # Hz
wp1 = 1.0 # Hz
wp2 = 35 # Hz
ws2 = 50 # Hz
 
# plantilla normalizada a Nyquist en dB
frecs = np.array([0.0,         ws1,         wp1,     wp2,     ws2,         nyq_frec   ]) / nyq_frec
gains = np.array([-atenuacion, -atenuacion, -ripple, -ripple, -atenuacion, -atenuacion])
# convertimos a veces para las funciones de diseño
gains = 10**(gains/20)

#Para el remez cascadeado
numtaps_hp = 2501 
trans_width = 0.7
band = [1, 35]  
edges_hp = [0, band[0] - trans_width , band[0],  0.5*fs_ecg]
desired_hp = [0.01,1] 
num_hp = sig.remez(numtaps_hp, edges_hp, desired_hp, fs=fs_ecg)
numtaps_lp = 1001
trans_width = 5
edges_lp = [0, wp2, wp2+trans_width, 0.5*fs_ecg]
desired_lp = [1, 0.007]
num_lp = sig.remez(numtaps_lp, edges_lp, desired_lp, fs=fs_ecg)
num_cascado = np.convolve(num_hp, num_lp)


#Diseño de filtros

bp_sos_butter = sig.iirdesign(wp=np.array([wp1, wp2]) / nyq_frec, ws=np.array([ws1, ws2]) / nyq_frec, gpass=0.5, gstop=40., analog=False, ftype='butter', output='sos')
bp_sos_cheby = sig.iirdesign(wp=np.array([wp1, wp2]) / nyq_frec, ws=np.array([ws1, ws2]) / nyq_frec, gpass=0.5, gstop=40., analog=False, ftype='cheby1', output='sos')
bp_sos_cauer = sig.iirdesign(wp=np.array([wp1, wp2]) / nyq_frec, ws=np.array([ws1, ws2]) / nyq_frec, gpass=0.5, gstop=40., analog=False, ftype='ellip', output='sos')

cant_coef = 3001 #para ventanas

gains = 10**(gains/20)

num_win =   sig.firwin2(cant_coef, frecs, gains , window ='hann' )

den = 1.0



# Filtros IIR con fase cero
ECG_f_butt  = sosfiltfilt(bp_sos_butter, ecg_one_lead.flatten())
ECG_f_cheby = sosfiltfilt(bp_sos_cheby, ecg_one_lead.flatten())
ECG_f_cauer = sosfiltfilt(bp_sos_cauer, ecg_one_lead.flatten())

# Filtros FIR con fase cero

ECG_f_remez = filtfilt(num_cascado, den, ecg_one_lead.flatten())
ECG_f_win   = filtfilt(num_win,   den, ecg_one_lead.flatten())

estilos = {
    'original': {'color': 'tab:blue', 'linestyle': '-', 'label': 'ECG original', 'linewidth': 2},
    'butter':   {'color': 'tab:orange', 'linestyle': '--', 'label': 'Filtro Butterworth'},
    'cheby':    {'color': 'tab:green', 'linestyle': '-.', 'label': 'Filtro Chebyshev I'},
    'cauer':    {'color': 'tab:red', 'linestyle': ':', 'label': 'Filtro Cauer (elíptico)'},
    'hann':     {'color': 'tab:purple', 'linestyle': '--', 'label': 'FIR Window (Hann)'},
    'remez':    {'color': 'tab:cyan', 'linestyle': '-.', 'label': 'FIR Parks-McClellan'}
}

regs_interes = ([4000, 5500],[4100, 4600],[10_000, 11_000],
        )

for ii in regs_interes:
    # Intervalo acotado dentro del rango de datos
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([N, ii[1]]), dtype='uint')

    plt.figure()
    plt.plot(zoom_region, ecg_one_lead[zoom_region],**estilos['original'])
    # Solo se muestra el filtro FIR para evaluar inocuidad (puedes cambiar a Butterworth si prefieres)
    plt.plot(zoom_region, ECG_f_butt[zoom_region],  **estilos['butter'])
    plt.plot(zoom_region, ECG_f_cheby[zoom_region], **estilos['cheby'])
    plt.plot(zoom_region, ECG_f_cauer[zoom_region], **estilos['cauer'])
    plt.plot(zoom_region, ECG_f_win[zoom_region],**estilos['hann'])
    plt.plot(zoom_region, ECG_f_remez[zoom_region],  **estilos['remez'])

    plt.title(f'Filtrado en región sin interferencias: muestras {int(ii[0])} a {int(ii[1])}')
    plt.ylabel('Amplitud')
    plt.xlabel('Muestras (#)')
    plt.legend()
    plt.yticks([])
    plt.show()
    
regs_interes = (
        np.array([5, 5.2]) *60*fs_ecg, # minutos a muestras
        np.array([12, 12.4]) *60*fs_ecg, # minutos a muestras
        np.array([15, 15.2]) *60*fs_ecg, # minutos a muestras
        )

for ii in regs_interes:
    # Intervalo acotado dentro del rango de datos
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([N, ii[1]]), dtype='uint')

    plt.figure()
    plt.plot(zoom_region, ecg_one_lead[zoom_region],**estilos['original'])
    plt.plot(zoom_region, ECG_f_butt[zoom_region],  **estilos['butter'])
    plt.plot(zoom_region, ECG_f_cheby[zoom_region], **estilos['cheby'])
    plt.plot(zoom_region, ECG_f_cauer[zoom_region], **estilos['cauer'])
    plt.plot(zoom_region, ECG_f_win[zoom_region],**estilos['hann'])
    plt.plot(zoom_region, ECG_f_remez[zoom_region],  **estilos['remez'])

    plt.title(f'Filtrado en región sin interferencias: muestras {int(ii[0])} a {int(ii[1])}')
    plt.ylabel('Amplitud')
    plt.xlabel('Muestras (#)')
    plt.legend()
    plt.yticks([])
    plt.show()