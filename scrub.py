import sys
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

audio = 'audio.wav'
x, sr = librosa.load(audio)
librosa.display.waveshow(x, sr=sr)
#amplitude vs. time

X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize = (10, 5))
librosa.display.specshow(Xdb, sr = sr, x_axis = 'time', y_axis = 'hz')
plt.colorbar()
#spectrogram

n_fft = 2048
plt.figure(figsize=(12, 4))
ft = np.abs(librosa.stft(x[:n_fft], hop_length = n_fft+1))
plt.plot(ft);
plt.title('Spectrum');
plt.xlabel('Frequency Bin');
plt.ylabel('Amplitude');
#FFT

plt.show()
# plt.savefig('\spec.png')
# input("Press Enter to continue...")