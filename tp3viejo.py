#%% módulos y funciones a importar
import numpy as np
import matplotlib.pyplot as plt

#%% Función para generar la señal seno
def seno(vmax, dc, ff, ph, N, fs):
    ts = 1/fs  # Paso de tiempo
    tiempoint = N * ts  # Intervalo total de tiempo
    tt = np.arange(start = 0, stop = tiempoint, step = ts)
    xx = vmax * np.sin( 2 * np.pi * ff * tt ) + dc
    return tt, xx

vmax = np.sqrt(2)
fs = 1000 # frecuencia de muestreo (Hz)
N = 1000  # cantidad de muestras
ts = 1/fs # tiempo de muestreo
df =  fs/N# resolución espectral
tiempoint = N * ts
dc = 0
ff = 250
ff2 = 700 #frecuencia mayor a nyquist

fsALTA = 100000  # frecuencia mucho mayor para simular continuo
NALTO = int(tiempoint * fsALTA) 
ttALTO = np.linspace(0, tiempoint, NALTO)
xx2_continua = vmax * np.sin(2 * np.pi * ff2 * ttALTO) + dc

tt, xx = seno(vmax, dc, ff, 0, N, fs)
tt, xx2 = seno(vmax, dc, ff2, 0, N, fs)

#%% Datos de la simulación
# Datos del ADC
B = 4 # bits

Vf = 2

q =  Vf/2**(B-1)# paso de cuantización de q Volts

Pq = (q ** 2) / 12 # Watts 

kn = 1 # escala de la potencia de ruido analógico

Pn = Pq * kn

nn = np.random.normal(0, np.sqrt(Pn), N)

sr = xx + nn  # Señal con ruido

sralias = xx2 + nn #señal con alias




#%% Visualización de resultados

# cierro ventanas anteriores
plt.close('all')

##################
# Señal temporal
##################
plt.figure(1)

# plt.plot(tt, sr, lw=1, color='blue', ls = 'dotted',  markersize=1, label='Señal analog de frecuencia=250')
plt.plot(tt, sralias, lw=1, color='green', ls = 'dotted',  markersize=1, label='Señal analog de frecuencia=700')
plt.plot(tt_alta, xx2_continua, lw=1, color='red', label='Señal continua f=700 Hz')

plt.title('Señales muestreadas para kn=1; B=4')
plt.xlabel('Tiempo [segundos]')
plt.ylabel('Amplitud [V]')
axes_hdl = plt.gca()
axes_hdl.legend()
plt.xlim(0, 0.005)
plt.show()


plt.figure(1)
######
# Espectro
###########

plt.figure(2)
ft_SR = 1/N*np.fft.fft( sr)
ft_SRA = 1/N*np.fft.fft( sralias)


ft_Nn = 1/N*np.fft.fft( nn)

ff = np.linspace(0, (N-1)*df, N)

bfrec = ff <= fs/2 #hasta nyquist, por la simetría de la fft, lo demás es redundante.

nNn_mean = np.mean(np.abs(ft_Nn)**2)

plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_SR[bfrec])**2), color='green', ls = 'dotted', label='Señal de frecuencia f = 250 Hz $' )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_SRA[bfrec])**2), color='blue', ls = 'dotted', label='Señal de frecuencia f = 300 Hz $' )


plt.title('Señal muestreada')
plt.ylabel('Densidad de Potencia [dB]')
plt.xlabel('Frecuencia [Hz]')
# %%
plt.ylim([-80, 40]) 


axes_hdl = plt.gca()
axes_hdl.legend()
