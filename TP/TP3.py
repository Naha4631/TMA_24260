import numpy as np
import matplotlib.pyplot as plt
import librosa
import IPython.display as ipd
from skimage import data, color


# PARTIE 1 : AUDIO ET REPLIEMENT (ALIASING)


# 1. Charger un fichier audio standard
y, sr = librosa.load(librosa.ex('trumpet'), sr=44100)

# 2. Sous-échantillonnage sauvage (on garde 1 échantillon sur 10)
facteur = 10
y_sous_echantillonne = y[::facteur]
sr_sous_echantillonne = sr // facteur

print("\n▶️ ÉCOUTEZ LE SIGNAL ORIGINAL (44.1 kHz) :")
display(ipd.Audio(y, rate=sr))

print("\n▶️ ÉCOUTEZ LE SIGNAL SOUS-ÉCHANTILLONNÉ (4.41 kHz) - Attention aux oreilles :")
display(ipd.Audio(y_sous_echantillonne, rate=sr_sous_echantillonne))

# 4. Affichage des Spectrogrammes
fig_audio, ax_audio = plt.subplots(1, 2, figsize=(15, 5))

# Spectrogramme Original
ax_audio[0].specgram(y, Fs=sr, NFFT=1024, cmap='magma')
ax_audio[0].set_title(f"Spectrogramme Original ({sr} Hz)")
ax_audio[0].set_ylabel("Fréquence (Hz)")
ax_audio[0].set_xlabel("Temps (s)")

# Spectrogramme Sous-échantillonné
nfft_reduit = 1024 // facteur
ax_audio[1].specgram(y_sous_echantillonne, Fs=sr_sous_echantillonne, NFFT=nfft_reduit, noverlap=nfft_reduit//2, cmap='magma')
# Le titre est maintenant sur une seule ligne sécurisée
ax_audio[1].set_title(f"Spectrogramme Sous-echantillonne ({sr_sous_echantillonne} Hz) - Aliasing !")
ax_audio[1].set_ylabel("Fréquence (Hz)")
ax_audio[1].set_xlabel("Temps (s)")

plt.tight_layout()
plt.show()


# PARTIE 2 : IMAGE ET QUANTIFICATION


# 1. Charger une image en niveaux de gris (8 bits)
image_couleur = data.astronaut()
image_grise = color.rgb2gray(image_couleur) 
image_8bits = (image_grise * 255).astype(np.uint8) 

# 2. Réduction du nombre de niveaux de gris
image_4_niveaux = (image_8bits // 64) * 64 
image_2_niveaux = (image_8bits // 128) * 128 

# 4. Pixélisation (Réduction de la résolution spatiale par 8)
facteur_spatial = 8
image_reduite = image_8bits[::facteur_spatial, ::facteur_spatial]

# Affichage des images
fig_img, ax_img = plt.subplots(2, 2, figsize=(12, 12))

ax_img[0,0].imshow(image_8bits, cmap='gray', vmin=0, vmax=255)
ax_img[0,0].set_title("1. Originale (8 bits)")

ax_img[0,1].imshow(image_4_niveaux, cmap='gray', vmin=0, vmax=255)
ax_img[0,1].set_title("2. Quantifiee a 2 bits (Banding)")

ax_img[1,0].imshow(image_2_niveaux, cmap='gray', vmin=0, vmax=255)
ax_img[1,0].set_title("3. Quantifiee a 1 bit (Binaire)")

ax_img[1,1].imshow(image_reduite, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
ax_img[1,1].set_title(f"4. Pixelisee (Divisee par {facteur_spatial})")

for a in ax_img.ravel():
    a.axis('off')

plt.tight_layout()
plt.show()