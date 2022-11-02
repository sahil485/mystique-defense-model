import sys
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

def show_plots(path, name):
    audio = path
    x, sr = librosa.load(audio, sr = 750)
    print("SR: {}".format(sr))
    librosa.display.waveshow(x, sr=sr)
    plt.title("Waveform: {}".format(path))
    plt.savefig('graphs/waveform/{}.png'.format(name))
    #amplitude vs. time

    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize = (10, 5))
    librosa.display.specshow(Xdb, sr = sr, x_axis = 'time', y_axis = 'hz')
    plt.colorbar()
    plt.title("Spectrogram: {}".format(path))
    plt.savefig('graphs/spec/{}.png'.format(name))
    #spectrogram

    n_fft = 2048
    plt.figure(figsize=(12, 4))
    ft = np.abs(librosa.stft(x[:n_fft], hop_length = n_fft+1))
    plt.plot(ft);
    plt.title('Spectrum');
    plt.xlabel('Frequency Bin');
    plt.ylabel('Amplitude');
    plt.title("FFT: {}".format(path))
    plt.savefig('graphs/FFT/{}.png'.format(name))
    #FFT

    plt.show()
    # plt.savefig('\spec.png')
    # input("Press Enter to continue...")

ria = "ria"
sahil = "sahil"
anshul = "anshul"
alex = 'alex'

ria_path = 'recordings/ria.m4a'
sahil_path = 'recordings/sahil.m4a'
anshul_path = 'recordings/anshul.m4a'
alex_path = 'recordings/alex.m4a'

if __name__ == "__main__":
    show_plots(ria_path, ria)
    show_plots(sahil_path, sahil)
    show_plots(anshul_path, anshul)
    show_plots(alex_path, alex)
