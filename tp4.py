import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

#%% Función para generar la señal seno
def seno(N, R, SNR):
    
    N=1000
    fs=1000
    df= fs/N
    w0 = fs/4
    a1=np.sqrt(2)
    R=200
    

    t = np.arange(0,1,1/N).reshape(N, 1)
    tt=np.tile(t, (1,R))#repite tt 200 veces en el eje de las columnas


    fr = np.random.uniform(-1/2, 1/2, size=(1, R))
    w1 = w0 + fr * df

    S = a1*np.sin(2*np.pi*w1*tt) #1000 * 200
    
    potseñal = np.mean(S**2)
    potruido = potseñal / (10**(SNR/10))
    

    ruido=np.random.normal(0, np.sqrt(potruido), size=S.shape)
    
    Xr = S + ruido #genera 200 señales con ruido
    
    return Xr, w1

#%%señales
Xr, w1 = seno(N=1000, R=200, SNR=10)
fs = 1000
N=1000
df=fs/N
ff = np.linspace(0, (N-1)*df, N)
bfrec = ff <= fs/2 #hasta nyquist, por la simetría de la fft, lo demás es redundante.
w0 = fs/4

Xr_fft = np.fft.fft(Xr, axis=0)
MAG_Xr_fft = np.abs(Xr_fft)

    
#%% VENTANAS



ventana2=sig.windows.blackmanharris(N).reshape(N,1)
senal_BH = Xr * ventana2 #1000X200

senal_BH_fft= np.fft.fft(senal_BH,axis=0)
MAGsenal_BH_fft=np.abs(senal_BH_fft) 

ventana=sig.windows.flattop(N).reshape(N,1)
senal_FT = Xr * ventana #1000X200

senal_FT_fft= np.fft.fft(senal_FT,axis=0)
MAGsenal_FT_fft=np.abs(senal_FT_fft) 

plt.figure(figsize=(10, 5))  # Ajustar tamaño del gráfic

plt.plot(ff[bfrec], 10* np.log10(2*(MAGsenal_BH_fft[bfrec,:])**2))
plt.title("Espectro de la primera realización (magnitud de la FFT)")
plt.xlabel("Frecuencia (bin)")
plt.ylabel("Magnitud")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))

plt.plot(ff[bfrec], 10* np.log10(2*(MAGsenal_FT_fft[bfrec,:])**2))
plt.title("Espectro de la primera realización (magnitud de la FFT)")
plt.xlabel("Frecuencia (bin)")
plt.ylabel("Magnitud")
plt.grid(True)
plt.show()

aest_BH = MAGsenal_BH_fft[round(w0)+1]
aest_FT = MAGsenal_FT_fft[round(w0)+1]
aest_rect = MAG_Xr_fft[round(w0)+1]

#############
# Histograma
#############

plt.figure(figsize=(10,5))
bins = 30
plt.hist(aest_rect, bins=bins, alpha=0.5, label="Rectangular")
plt.hist(aest_BH, bins=bins, alpha=0.5, label="Blackman-Harris")
plt.hist(aest_FT, bins=bins, alpha=0.5, label="Flat Top")

plt.xlabel("Valor del estimador")
plt.ylabel("Frecuencia")
plt.title("Comparación de histogramas de los estimadores de la magnitud en w₀ = 250 Hz")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

