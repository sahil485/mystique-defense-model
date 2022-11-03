import sys
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

def show_plots(path, name):
    audio = path
    x, sr = librosa.load(audio, sr = 800)
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

    plt.show()

ria = "ria"
sahil = "sahil"
anshul = "anshul"
alex = 'alex'
dan = 'dan'

ria_path = 'recordings/ria/ria1.m4a'
sahil_path = 'recordings/sahil/sahil1.m4a'
anshul_path = 'recordings/anshul/anshul1.m4a'
alex_path = 'recordings/alex/alex1.m4a'
dan_path = 'recordings/dan/dan1.m4a'

if __name__ == "__main__":
    show_plots(ria_path, ria)
    show_plots(sahil_path, sahil)
    show_plots(anshul_path, anshul)
    show_plots(alex_path, alex)
    show_plots(dan_path, dan)
