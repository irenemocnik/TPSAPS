# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 09:48:01 2025

@author: iremo
"""
import numpy as np
import matplotlib.pyplot as plt

N = 1000  #cantidad de datos
Fs = 1000 #frecuencia de muestreo
vect_temp = np.arange(0.0,N/Fs, 1/Fs)
#Intervalo de 0 a N/Fs=N*tiempototal con un paso de 1/Fs=Ts

vect_frec = np.arange(0.0, Fs/2.0, N/Fs)
#Desde 0 hasta la frec de Nyquist, con paso N/Fs=cantdedatos/frectotal=pasofrecuencial

canales_ADC = 8 
mat_datos = np.random.normal(0,1.0,size = [N,canales_ADC])
#simulo que el adc tiene las señales abiertas y pescan ruido en forma aleatoria
#pido que tome nuestras que siga una dist normal de media 0, desvest=1.
#El tamaño del vector de datos es 8 col de canales * 1000 filas de datos.

vector_bool = abs(mat_datos[:,0])>0.5
#asigna True o False a valores de muestras con vabs mayor a 0,5 en la columna 1.

indices = np.nonzero(vector_bool)[0]
#toma el vector_bool y me devuelve una lista con todos los no Falsos.

def algunafuncion (parametro1, parametro2):
    resultado=parametro1+parametro2
    return(resultado)

print(algunafuncion(3,8))


#%%
#TEST BENCH

#estructura de los plots:
def mi_testbench(sig_type):
    Ts= 1/Fs
    df=Fs/N
    
    tt= np.linspace(0, (N-1)*Ts, N).flatten()    
    ff= np.linspace(0, (N-1)*df,N).reshape(N,0)
    
    x= np.array([].dtype=np.float).reshape(N,0)
    tt=0
    
    plt.figure(1)
    line_hdls = plt-plot(tt,x)
    plt.title["Señal + sig_type['tipo']
