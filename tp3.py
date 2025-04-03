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
ff = 1

tt, xx = seno(vmax, dc, ff, 0, N, fs)

#%% Datos de la simulación
# Datos del ADC
B = 4 # bits
# B2 = 8
# B3 = 16

Vf = 2# rango simétrico de +/- Vf Volts

q =  Vf/2**(B-1)# paso de cuantización de q Volts
# q2 = Vf/2**(B2-1)
# q3 =Vf/2**(B3-1)


# datos del ruido (potencia de la señal normalizada, es decir 1 W)
Pq = (q ** 2) / 12 # Watts 
# Pq2 = (q2 ** 2) / 12
# Pq3 = (q3 ** 2) / 12

kn = 1 # escala de la potencia de ruido analógico
kn2 = 1/10
kn3 = 10

Pn = Pq * kn # 
# Pn2 = Pq2 * kn
# Pn3 = Pq3 * kn
Pn2 = Pq * kn2
Pn3 = Pq * kn3

# Señales

nn = np.random.normal(0, np.sqrt(Pn), N)
nn2 = np.random.normal(0, np.sqrt(Pn2), N)
nn3 = np.random.normal(0, np.sqrt(Pn3), N)


sr = xx + nn  # Señal con ruido
sr2 = xx + nn2  # Señal con ruido
sr3 = xx + nn3  # Señal con ruido

srq = q * np.round(sr / q)  # Señal cuantizada
srq2 = q * np.round(sr2 / q)
srq3 = q * np.round(sr3 / q)


nq = srq - sr  # Ruido de cuantización
nq2 = srq2 - sr2
nq3 = srq3 - sr3


#%% Visualización de resultados

# cierro ventanas anteriores
plt.close('all')

##################
# Señal temporal
##################
plt.figure(1)

plt.plot(tt, srq, lw=1, color='blue', ls = 'dotted', marker='.', markersize=1, markerfacecolor='blue', label='Señal cuantizada (kn=1)')
plt.plot(tt, srq2, lw=1, color='green', ls = 'dotted',  markersize=1, marker='.', label='Señal cuantizada (kn=1/10)')
plt.plot(tt, srq3, lw=1, color='red',  ls = 'dotted', marker='.',markersize=1,label=' Señal cuantizada (kn=10)')

plt.title('Señal muestreada para kn={1,1/10,10}')
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
axes_hdl = plt.gca()
axes_hdl.legend()
plt.show()

plt.figure(1)
######
# Espectro
###########

plt.figure(2)
ft_SR = 1/N*np.fft.fft( sr)


ft_Srq = 1/N*np.fft.fft( srq)
ft_Srq2 = 1/N*np.fft.fft( srq2)
ft_Srq3 = 1/N*np.fft.fft( srq3)

ft_SP = 1/N*np.fft.fft( xx)
ft_Nq = 1/N*np.fft.fft( nq)
ft_Nq2 = 1/N*np.fft.fft( nq2)
ft_Nq3 = 1/N*np.fft.fft( nq3)

ft_Nn = 1/N*np.fft.fft( nn)
ft_Nn2 = 1/N*np.fft.fft( nn2)
ft_Nn3 = 1/N*np.fft.fft( nn3)


# grilla de sampleo frecuencial
ff = np.linspace(0, (N-1)*df, N)

bfrec = ff <= fs/2 #hasta nyquist, por la simetría de la fft, lo demás es redundante.

Nnq_mean = np.mean(np.abs(ft_Nq)**2)
Nnq2_mean = np.mean(np.abs(ft_Nq2)**2)
Nnq3_mean = np.mean(np.abs(ft_Nq3)**2)

nNn_mean = np.mean(np.abs(ft_Nn)**2)
nNn2_mean = np.mean(np.abs(ft_Nn2)**2)
nNn3_mean = np.mean(np.abs(ft_Nn3)**2)

#plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_SP[bfrec])**2), color='orange', ls='dotted', label='$ s $ (Senoidal sin ruido)' )
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([nNn_mean, nNn_mean]) ),  color='red', ls='dotted', label= '$ \overline{n} = $' + '{:3.1f} dB'.format(10* np.log10(2* nNn_mean)) )
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([nNn2_mean, nNn2_mean]) ),  color='orange', ls='dotted', label= '$ \overline{n} = $' + '{:3.1f} dB'.format(10* np.log10(2* nNn2_mean)) )
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([nNn3_mean, nNn3_mean]) ),  color='magenta', ls='dotted', label= '$ \overline{n} = $' + '{:3.1f} dB'.format(10* np.log10(2* nNn3_mean)) )

#plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_SR[bfrec])**2), color='green', ls = 'dotted', label='$ s_R = s + n $' )
#plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Srq[bfrec])**2), color='blue',lw=1, label='$ s_Q = Q_{B,V_F}\{s_R\}$' )

plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([Nnq_mean, Nnq_mean]) ), '--r', label='$ \overline{n_Q} = $' + '{:3.1f} dB'.format(10* np.log10(2* Nnq_mean)) )
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([Nnq2_mean, Nnq2_mean]) ), '--o', label='$ \overline{n2_Q} = $' + '{:3.1f} dB'.format(10* np.log10(2* Nnq2_mean)) )
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([Nnq3_mean, Nnq3_mean]) ), '--m', label='$ \overline{n3_Q} = $' + '{:3.1f} dB'.format(10* np.log10(2* Nnq3_mean)) )

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
