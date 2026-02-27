import numpy as np
import matplotlib.pyplot as plt

# 1.1 Génération d’un signal sinusoïdal

f0 = 10      
A = 1        
phi = 0      
fs = 100     
duration = 1 

t = np.linspace(0, duration, int(fs * duration), endpoint=False)
x = A * np.sin(2 * np.pi * f0 * t + phi)

plt.figure(figsize=(10,4))
plt.plot(t, x)
plt.title("Signal sinusoïdal x(t)")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()



# 1.2 Ajout de bruit

bruit = np.random.normal(0, 0.3, len(x))  
y = x + bruit

plt.figure(figsize=(10,4))
plt.plot(t, x, label="Signal pur")
plt.plot(t, y, label="Signal bruité", alpha=0.7)
plt.title("Signal pur vs Signal bruité")
plt.xlabel("Temps (s)")
plt.legend()
plt.grid()
plt.show()



# 1.3 Signaux élémentaires et Convolution

rect = np.zeros(100)
rect[20:40] = 1

conv_result = np.convolve(rect, rect)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(rect)
plt.title("Signal Porte (Rect)")

plt.subplot(1,2,2)
plt.plot(conv_result)
plt.title("Convolution Rect * Rect")

plt.tight_layout()
plt.show()