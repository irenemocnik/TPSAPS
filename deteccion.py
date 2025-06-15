"""
@author: iremo
"""

import numpy as np
from scipy import signal as sig
from scipy.signal import correlate, find_peaks
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import sosfiltfilt, filtfilt

#%% Diseño del filtro

fs_ecg = 1000
nyq_frec = fs_ecg / 2


mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg = mat_struct['ecg_lead'].flatten()
hb1 = mat_struct['heartbeat_pattern1'].flatten()
hb2 = mat_struct['heartbeat_pattern2'].flatten()
qrs = mat_struct['qrs_pattern1'].flatten()

ecg = (ecg - np.mean(ecg)) / np.std(ecg)

# ws1 = 0.1 # Hz
# wp1 = 1.0 # Hz
# wp2 = 35 # Hz
# ws2 = 50 # Hz
# bp_sos_butter = sig.iirdesign(wp=np.array([wp1, wp2]) / nyq_frec, ws=np.array([ws1, ws2]) / nyq_frec, gpass=0.5, gstop=40., analog=False, ftype='butter', output='sos')
# ECG_f_butt  = sosfiltfilt(bp_sos_butter, ecg.flatten())


# hb1 =( hb1 - np.mean(hb1)) / np.std(hb1)
# hb2 =( hb2 - np.mean(hb2)) / np.std(hb2)
qrs =( qrs - np.mean(qrs)) / np.std(qrs)

#un vector de tiempo para cada patron
t_hb1 = np.arange(len(hb1)) / fs_ecg
t_hb2 = np.arange(len(hb2)) / fs_ecg
t_qrs = np.arange(len(qrs)) / fs_ecg

t_ecg = np.arange(len(ecg)) / fs_ecg


plt.figure(figsize=(12, 4))
plt.plot(t_qrs, qrs)
plt.title('Patrón QRS usado para detección')
plt.xlabel('Tiempo [s]')
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(t_ecg[:3000], ecg[:3000])
plt.title('Primeros 3 segundos del ECG')
plt.xlabel('Tiempo [s]')
plt.grid(True)
plt.show()
#%% Detección por correlación

qrs_matched = qrs[::-1]
corr = np.convolve(ecg, qrs_matched, mode='same')
corr = (corr - np.mean(corr)) / np.std(corr)

peaks, _ = find_peaks(corr, height = 1.0, distance=fs_ecg*0.3)

plt.figure(figsize=(12, 4))
plt.plot(t_ecg, corr, label='Correlación (Matched Filter)')
plt.plot(t_ecg[peaks], corr[peaks], 'rx', label='Picos detectados')
plt.legend()
plt.grid(True)
plt.title("Señal de correlación y detección de latidos")
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(t_ecg, ecg, label='ECG')
plt.plot(t_ecg[peaks], ecg[peaks], 'rx', label='Picos detectados')
plt.legend()
plt.grid(True)
plt.title("Latidos detectados sobre el ECG")
plt.show()


#fraccion de detecciones verdaderas
VP = 0 #Verdaderos positivos (latidos reales detectados)
FP = 0 # deteccion sin latido
FN = 0 # falsos negativos (latidos reales que no fueron detectados)

qrsVerdaderos = mat_struct['qrs_detections'].flatten()
tolerancia = 0.05  # segundos
ventana = int(tolerancia * fs_ecg)


latidos_usados = np.zeros(len(qrsVerdaderos), dtype=bool)

for pico in peaks:
    acierto = False
    for i, real in enumerate(qrsVerdaderos):
        if not latidos_usados[i] and abs(pico - real) <= ventana:
            VP = VP + 1
            latidos_usados[i] = True  # Marcamos ese latido como usado
            acierto = True
            break  # Ya está, no sigas buscando
    if not acierto:
        FP = FP + 1
        
FN = np.sum(latidos_usados == False)


fracTotal = VP / (VP + FN)
fracDetectados = VP / (VP + FP)

print(f"Latidos detectados correctamente: {VP}")
print(f"Latidos detectados que no lo eran): {FP}")
print(f"Latidos verdaderos no detectados: {FN}")
print(f"Fracción de latidos detectados (sensibilidad): {fracTotal:.2f}")
print(f"Fraccion de latidos detectados que eran verdaderos latidos (Valor Predictivo Positivo (PPV)): {fracDetectados:.2f}")