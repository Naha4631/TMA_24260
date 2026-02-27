import os
import subprocess
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from IPython.display import Audio, display

# 1. Verification of the file
target_video = "projet.mp4"
if not os.path.exists(target_video):
    print(f"ERROR: '{target_video}' not found in the sidebar!")
    print("Please upload your video and rename it exactly to 'projet.mp4'")
else:
    # 2. Audio Extraction
    audio_out = "audio_principal.wav"
    if not os.path.exists(audio_out):
        subprocess.run(f"ffmpeg -i {target_video} -vn -acodec pcm_s16le -ar 44100 -ac 1 {audio_out} -y", shell=True)

    # 3. Processing and Display
    if os.path.exists(audio_out):
        y, sr = librosa.load(audio_out, sr=None)
        
        # Display Original Audio
        display(Audio(y, rate=sr))

        # FFT Analysis
        y1 = y[0:sr*5]
        Y_fft = np.fft.rfft(y1)
        freqs = np.fft.rfftfreq(len(y1), 1/sr)
        
        plt.figure(figsize=(10, 3))
        plt.plot(freqs, np.abs(Y_fft))
        plt.xlim(0, 4000)
        plt.show()

        # ICA Separation
        y2 = y[sr*5:sr*10]
        length = min(len(y1), len(y2))
        s1, s2 = y1[:length], y2[:length]
        
        X = np.c_[0.5*s1 + 0.5*s2, 0.7*s1 + 0.3*s2]
        ica = FastICA(n_components=2, random_state=42)
        S_recovered = ica.fit_transform(X)

        # Display Separated Results
        display(Audio(S_recovered[:, 0], rate=sr))
        display(Audio(S_recovered[:, 1], rate=sr))

        plt.figure(figsize=(10, 4))
        plt.subplot(2,1,1); plt.plot(S_recovered[:,0])
        plt.subplot(2,1,2); plt.plot(S_recovered[:,1])
        plt.show()

        # Matricule(24092, 24260)


# 4) Analyse
#   L’analyse du signal audio extrait du fichier vidéo montre clairement les propriétés d’un signal vocal réel.
#    Dans le domaine temporel, le signal présente une variation non stationnaire avec des zones d’énergie faible (pauses) et des zones d’énergie élevée correspondant aux segments parlés. L’amplitude varie de manière significative, ce qui confirme la dynamique naturelle de la voix humaine.
#    L’analyse fréquentielle par Transformée de Fourier (FFT), appliquée sur les 5 premières secondes du signal, met en évidence une forte concentration d’énergie dans les basses fréquences (principalement en dessous de 1000 Hz). On observe un pic dominant dans la zone des fréquences basses, ainsi que des composantes secondaires correspondant aux harmoniques.
#    La décroissance progressive de l’énergie au-delà de 1500 Hz confirme qu’il s’agit d’un signal vocal et non d’un bruit blanc ou d’un signal musical large bande.
#    Ainsi, l’analyse temporelle et fréquentielle confirme que le signal étudié est cohérent avec un enregistrement vocal naturel.

# 5) Superposition

# Dans la deuxième partie, une simulation de mélange a été réalisée en combinant deux segments du signal :
# Segment 1 : premières 5 secondes
# Segment 2 : secondes 5 à 10
# Un mélange linéaire a été construit sous forme matricielle afin de simuler deux capteurs recevant des combinaisons différentes des deux sources.
# L’algorithme FastICA a ensuite été appliqué pour effectuer une séparation aveugle des sources.
# Les résultats obtenus montrent que :
# Deux signaux distincts ont été reconstruits.
# Les formes d’onde récupérées présentent des variations temporelles différentes.
# La séparation n’est pas parfaitement nette mais reste significative.
# Cette séparation partielle s’explique par le fait que les deux segments proviennent du même locuteur, ce qui réduit l’indépendance statistique entre les sources. Malgré cela, l’algorithme réussit à extraire deux composantes dominantes distinctes.


# 6) Étude des caractéristiques

# L’étude des caractéristiques du signal permet d’identifier :
# Une structure harmonique visible dans le spectre.
# Une concentration d’énergie dans les basses et moyennes fréquences.
# Une forte variabilité temporelle liée à l’articulation.
# Une atténuation progressive des hautes fréquences.
# Les signaux reconstruits par ICA présentent également des différences d’amplitude et de dynamique, indiquant que le mélange a modifié la distribution énergétique initiale.
# Le signal ne présente pas de composantes bruitées importantes, ce qui confirme la bonne qualité de l’enregistrement.
# Globalement, les caractéristiques observées sont typiques d’un signal vocal humain enregistré en environnement relativement stable.

# 7) Synthèse

# L’ensemble du projet met en évidence les principes fondamentaux du traitement du signal audio.
# L’analyse fréquentielle a permis d’identifier les composantes dominantes du signal vocal.
# La simulation de mélange a reproduit une situation réaliste de superposition de sources.
# L’application de l’algorithme ICA a démontré la possibilité de séparer partiellement des signaux combinés.
# Bien que la séparation ne soit pas parfaite, les résultats obtenus sont cohérents avec la théorie, notamment en raison de la similarité statistique des sources utilisées.
# Le projet valide donc de manière satisfaisante les concepts étudiés : analyse spectrale, mélange linéaire et séparation aveugle de sources.