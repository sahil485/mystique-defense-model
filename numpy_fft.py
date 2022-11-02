import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt
import librosa
import librosa.display

def mod_plots(name):
    sr = 800
    ts = 1/sr
    t = np.arange(0, 1, ts)

    x, sr = librosa.load('recordings/{}.m4a'.format(name), sr = sr)
    print(x[:400])
    print("\n")
    print(x[400:1200])
    arr = np.where(x >= 0.01)
    print(arr)
    print(type(arr))
    # print(x[arr[0]])
    # print(x[arr[1]])
    # print(x[arr[2]])

    # librosa.display.waveshow(x[arr[0][0]:], sr=sr)
    librosa.display.waveshow(x[arr[0][0]:arr[0][-1]], sr = sr)
    plt.title("Waveform: {}".format(name))
    plt.savefig('graphs/waveform/{}_threshold.png'.format(name))

    x = x[arr[0][0]:arr[0][-1]]

    vals = fft(x)
    length = len(vals)
    n = np.arange(length)
    T = length/sr

    freq = n/T

    plt.figure(figsize = (12, 6))
    plt.subplot(121)

    plt.stem(freq, np.abs(vals), 'b', \
             markerfmt=" ", basefmt="-b")
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')
    plt.xlim(0, 300)
    plt.savefig('graphs/FFT/{}.png'.format(name))
    plt.show()
    print(vals)

if __name__ == "__main__":
    mod_plots('sahil')
    mod_plots("ria")
    mod_plots("alex")
    mod_plots("anshul")