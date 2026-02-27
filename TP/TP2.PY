import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display


# 2.1 Analyse spectrale d’un son pur

fs = 44100        
duration = 2      
t = np.linspace(0, duration, int(fs*duration), endpoint=False)


f1 = 440   
f2 = 880   

signal = np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t)

print("Signal original")
display(Audio(signal, rate=fs))

S = np.fft.fft(signal)
freqs = np.fft.fftfreq(len(signal), 1/fs)


plt.figure(figsize=(10,4))
plt.plot(freqs, np.abs(S))
plt.xlim(0,2000)
plt.title("Spectre du signal original")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()


# 2.2 Identification et Filtrage d’un bruit

f_bruit = 5000
bruit = 0.3 * np.sin(2*np.pi*f_bruit*t)

signal_bruite = signal + bruit

print("Signal avec bruit")
display(Audio(signal_bruite, rate=fs))

S_bruit = np.fft.fft(signal_bruite)

plt.figure(figsize=(10,4))
plt.plot(freqs, np.abs(S_bruit))
plt.xlim(0,6000)
plt.title("Spectre du signal bruité")
plt.xlabel("Fréquence (Hz)")
plt.grid()
plt.show()


fc = 2000   
H = np.zeros_like(S_bruit)

for i in range(len(freqs)):
    if abs(freqs[i]) < fc:
        H[i] = 1

S_filtre = S_bruit * H


signal_filtre = np.fft.ifft(S_filtre)
signal_filtre = np.real(signal_filtre)

print("Signal après filtrage")
display(Audio(signal_filtre, rate=fs))


plt.figure(figsize=(10,4))
plt.plot(freqs, np.abs(S_filtre))
plt.xlim(0,6000)
plt.title("Spectre après filtrage")
plt.xlabel("Fréquence (Hz)")
plt.grid()
plt.show()