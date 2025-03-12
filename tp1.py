# -*- coding: utf-8 -*-




import numpy as np
import matplotlib.pyplot as plt


fs = 20000000  # 20MHz (Más puntos, más suave)
f0 = 10000  # Frecuencia base

N = int(fs / f0)  # Muestras por ciclo

# Aumento del tiempo total mostrado (duplicando el número de ciclos)
t = np.linspace(0, 20 * (N - 1) / fs, 20* N)  # *20 Para visualizar más ciclos
pasofrec = fs / N  # Intervalo de frecuencias

def mi_funcion_sen(vmax, dc, ff, ph, fs, t):
    """
    Esto genera una señal senoidal parametrizable cuyos parámetros son:
    vmax: Amplitud máxima de la señal.
    dc: Valor medio de la señal.
    ff: Frecuencia de la señal en Hz.
    ph: Fase de la señal en radianes.
    fs: Frecuencia de muestreo en Hz.
    t: Vector de tiempos predefinido.
    La señal devuelve x, vector de valores de la función de tamaño Nx1
    """
    x = vmax * np.sin(2 * np.pi * ff * t + ph) + dc
    return x.reshape(-1, 1)
    #Genera una señal senoidal, el reshape asegura que el vector x sea Nx1.

# def funcion_cuadrada(amplitud,
#     """
#    Genera una señal cuadrada de parámetros:
#     amplitud: Amplitud de la señal.
#     dc: Valor medio
#     ff: Frecuencia [Hz]
   
#     t: Vector de tiempos predefinido.
#     La señal devuelve xcuadrada, vector de valores de la función de tamaño Nx1
#     """
#     xcuadrada = np.sign(x)
#     return tt, xx

# Parámetros generales
frecuencias = [500, 999, 1001, 2001]  # Frecuencias solicitadas
#duracion_pulso = 0.001  # Duración del pulso [s]

# Generación de señales
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# Gráfico de Señales senoidales
for f in frecuencias:
    x = mi_funcion_sen(vmax=1, dc=0, ff=f, ph=0, fs=fs, t=t)
    axs[0].plot(t, x, label=f"f = {f} Hz")

axs[0].set_xlabel("Tiempo [s]")
axs[0].set_ylabel("Amplitud")
axs[0].set_title("Señales senoidales con diferentes frecuencias")
axs[0].legend()
axs[0].grid()

# Gráfico de Señales pulso
# for f in frecuencias:
#     x_pulso = señal_pulso(amplitud=1, dc=0, ff=f, duracion_pulso=duracion_pulso, t=t)
#     axs[1].plot(t, x_pulso, label=f"f = {f} Hz")

# axs[1].set_xlabel("Tiempo [s]")
# axs[1].set_ylabel("Amplitud")
# axs[1].set_title("Señales de pulso con diferentes frecuencias")
# axs[1].legend()
# axs[1].grid()

plt.tight_layout()
plt.show()




   