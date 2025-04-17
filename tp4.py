import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

#%% Función para generar la señal seno
def seno(N, R, SNR):
    fs=1000
    df= fs/N
    w0 = fs/4
    a1=np.sqrt(2)

    t = np.arange(0,1,1/N).reshape(N, 1)
    tt=np.tile(t, (1,R))

    fr = np.random.uniform(-1/2, 1/2, size=(1, R))
    w1 = w0 + fr * df

    S = a1*np.sin(2*np.pi*w1*tt)
    potseñal = np.mean(S**2)
    potruido = potseñal / (10**(SNR/10))

    ruido=np.random.normal(0, np.sqrt(potruido), size=S.shape)
    Xr = S + ruido

    return Xr, w1

#%% Señales
N = 1000
R = 200
SNR = 10
fs = 1000
df = fs / N
Xr, w1 = seno(N=N, R=R, SNR=SNR)

ff = np.linspace(0, (N-1)*df, N)
bfrec = ff <= fs/2
w0 = fs/4
a1= np.sqrt(2)

Xr_fft = np.fft.fft(Xr, axis=0)
MAG_Xr_fft = 1/N * np.abs(Xr_fft)

ventana2 = sig.windows.blackmanharris(N).reshape(N,1)
senal_BH = Xr * ventana2
senal_BH_fft= np.fft.fft(senal_BH,axis=0)
MAGsenal_BH_fft = 1/N * np.abs(senal_BH_fft) 

ventana = sig.windows.flattop(N).reshape(N,1)
senal_FT = Xr * ventana
senal_FT_fft= np.fft.fft(senal_FT,axis=0)
MAGsenal_FT_fft = 1/N * np.abs(senal_FT_fft) 

#%% FFT (una realización)
plt.figure(figsize=(10, 5))
plt.plot(ff[bfrec], 10 * np.log10(2 * (MAG_Xr_fft[bfrec, 0])**2), label="Rectangular")
plt.plot(ff[bfrec], 10 * np.log10(2 * (MAGsenal_BH_fft[bfrec, 0])**2), label="Blackman-Harris")
plt.plot(ff[bfrec], 10 * np.log10(2 * (MAGsenal_FT_fft[bfrec, 0])**2), label="Flat Top")
plt.title("FFT de la 1era realización")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud (dB)")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(ff[bfrec], 10 * np.log10(2 * (MAG_Xr_fft[bfrec, 0])**2), label="Rectangular")
plt.plot(ff[bfrec], 10 * np.log10(2 * (MAGsenal_BH_fft[bfrec, 0])**2), label="Blackman-Harris")
plt.plot(ff[bfrec], 10 * np.log10(2 * (MAGsenal_FT_fft[bfrec, 0])**2), label="Flat Top")
plt.xlim(240, 260)
plt.title("Zoom FFT de la 1era realización")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud (dB)")
plt.grid(True)
plt.legend()
plt.show()

#%% Estimadores de magnitud
aest_RECT = np.zeros(R)
aest_BH = np.zeros(R)
aest_FT = np.zeros(R)

for i in range(R):
    k_real = int(np.round(w1[0, i]))  # bin más cercano a la frecuencia real
    aest_RECT[i] = MAG_Xr_fft[k_real, i]
    aest_BH[i] = MAGsenal_BH_fft[k_real, i]
    aest_FT[i] = MAGsenal_FT_fft[k_real, i]

#%% Calculo de sesgo y varianza (magnitud)
sesgo_RECT = np.mean(aest_RECT) - a1
sesgo_BH = np.mean(aest_BH) - a1
sesgo_FT = np.mean(aest_FT) - a1

var_RECT = np.var(aest_RECT)
var_BH = np.var(aest_BH)
var_FT = np.var(aest_FT)

#%% Histograma de magnitud
plt.figure(figsize=(10,5))
bins = 30
plt.hist(aest_RECT, bins=bins, alpha=0.5, label=f"Rectangular\nSesgo: {sesgo_RECT:.4f}\nVar: {var_RECT:.4f}")
plt.hist(aest_BH, bins=bins, alpha=0.5, label=f"Blackman-Harris\nSesgo: {sesgo_BH:.4f}\nVar: {var_BH:.4f}")
plt.hist(aest_FT, bins=bins, alpha=0.5, label=f"Flat Top\nSesgo: {sesgo_FT:.4f}\nVar: {var_FT:.4f}")
plt.xlabel("Valor del estimador")
plt.ylabel("Frecuencia")
plt.title("Comparación de histogramas de estimadores de magnitud en ω₀")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#%% Estimador de frecuencia
omega1est_RECT = np.zeros(R)
omega1est_BH = np.zeros(R)
omega1est_FT = np.zeros(R)

for i in range(R):
    kmax_RECT = np.argmax(MAG_Xr_fft[:N//2, i])
    kmax_BH = np.argmax(MAGsenal_BH_fft[:N//2, i])
    kmax_FT = np.argmax(MAGsenal_FT_fft[:N//2, i])
    
    omega1est_RECT[i] = kmax_RECT * df
    omega1est_BH[i] = kmax_BH * df
    omega1est_FT[i] = kmax_FT * df

# Cálculo de sesgo y varianza (frecuencia)
sesgo_w_RECT = np.mean(omega1est_RECT) - w0
sesgo_w_BH = np.mean(omega1est_BH) - w0
sesgo_w_FT = np.mean(omega1est_FT) - w0

var_w_RECT = np.var(omega1est_RECT)
var_w_BH = np.var(omega1est_BH)
var_w_FT = np.var(omega1est_FT)

# Histograma de estimadores de frecuencia
plt.figure(figsize=(10, 5))
plt.hist(omega1est_RECT, bins=30, alpha=0.5, label=f'Rectangular\nSesgo: {sesgo_w_RECT:.4f}\nVar: {var_w_RECT:.4f}')
plt.hist(omega1est_BH, bins=30, alpha=0.5, label=f'Blackman-Harris\nSesgo: {sesgo_w_BH:.4f}\nVar: {var_w_BH:.4f}')
plt.hist(omega1est_FT, bins=30, alpha=0.5, label=f'Flat Top\nSesgo: {sesgo_w_FT:.4f}\nVar: {var_w_FT:.4f}')
plt.axvline(w0, color='k', linestyle='--', label='Frecuencia real')
plt.xlabel('Frecuencia estimada [Hz]')
plt.ylabel('Frecuencia')
plt.title('Histograma de estimadores de frecuencia')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#%% ZERO PADDING
Npad = 4000
dfpad = fs / Npad
ffpad = np.linspace(0, fs - dfpad, Npad)

# Zero-padding de señales
Xrpad = np.zeros((Npad, R))
senal_BH_pad = np.zeros((Npad, R))
senal_FT_pad = np.zeros((Npad, R))

Xrpad[:N, :] = Xr
senal_BH_pad[:N, :] = senal_BH
senal_FT_pad[:N, :] = senal_FT

# FFT padded
MAG_RECT_pad = 1/N * np.abs(np.fft.fft(Xrpad, axis=0))
MAG_BH_pad = 1/N * np.abs(np.fft.fft(senal_BH_pad, axis=0))
MAG_FT_pad = 1/N * np.abs(np.fft.fft(senal_FT_pad, axis=0))

# Estimadores de frecuencia con padding
omega1est_RECT_pad = np.zeros(R)
for i in range(R):
    kmax = np.argmax(MAG_RECT_pad[:Npad//2, i])
    omega1est_RECT_pad[i] = kmax * dfpad

# Comparación de histogramas con y sin padding
plt.figure(figsize=(10, 5))
plt.hist(omega1est_RECT, bins=30, alpha=0.5, label='Rectangular (sin padding)')
plt.hist(omega1est_RECT_pad, bins=30, alpha=0.5, label='Rectangular (con padding)')
plt.axvline(w0, color='k', linestyle='--', label='Frecuencia real')
plt.xlabel('Frecuencia estimada [Hz]')
plt.ylabel('Frecuencia')
plt.title('Efecto del zero-padding en estimación de frecuencia')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Comparación visual en una realización
i = 0
plt.figure(figsize=(10,5))
plt.plot(ff, MAGsenal_BH_fft[:, i], label="Sin padding (Blackman-Harris)")
plt.plot(ffpad, MAG_BH_pad[:, i], label="Con padding (Blackman-Harris)", alpha=0.7)
plt.axvline(w0, color='k', linestyle='--', label='Frecuencia real')
plt.xlim(245, 255)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.title("FFT con y sin zero-padding (ventana BH)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


print(var_RECT)
print(var_BH)
print(var_FT)

print(sesgo_RECT)
print(sesgo_BH)
print(sesgo_FT)

print(var_w_RECT)
print(var_w_BH)
print(var_w_FT)

print(sesgo_w_RECT)
print(sesgo_w_BH)
print(sesgo_w_FT)


