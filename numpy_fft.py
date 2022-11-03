import numpy as np
from numpy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import librosa
import librosa.display

max_freq = dict()

def mod_plots(name):
    sr = 800
    ts = 1/sr
    t = np.arange(0, 1, ts)

    x, sr = librosa.load('recordings/{}.m4a'.format(name), sr = sr)
    arr = np.where(x >= 0.01) #thresholding the signal to find values that contain speech

    # librosa.display.waveshow(x[arr[0][0]:arr[0][-1]], sr = sr)
    # plt.title("Waveform: {}".format(name))
    # plt.savefig('graphs/waveform/{}_threshold.png'.format(name))

    offset = arr[0][0]

    x = x[arr[0][0]:arr[0][-1]]

    vals = fft(x) # FFT of signal
    length = len(vals)

    n = np.arange(length)
    T = length/sr

    freq_bins = fftfreq(len(vals))[np.argmax(np.abs(vals))] #finds the bin placement of the highest frequency
    MAX = np.abs(freq_bins*sr)
    max_freq[name] = MAX
    print("{} MAX: {}".format(name, MAX)) #most prevalent frequency

    freq = n/T
    # print("FREQ: {}".format(freq))

    # print(np.max(np.abs(vals)))
    m_ind = np.argmax(np.abs(vals))

    # print("max_freq: {}".format(m_ind))
    # print(" magnitude: {}".format(np.abs(vals[m_ind])))
    # vals[31] = 0

    m2_ind = np.where(vals == np.unique(vals)[-1])
    # print("max_freq2: {}".format(m2_ind[0][0]))

    v2 = np.zeros(length)
    v2[m_ind] = np.abs(vals[m_ind])
    v2[m2_ind] = np.abs(vals[m2_ind])

    plt.figure(figsize = (12, 6))
    # plt.subplot(121)

    # plt.stem(freq, np.abs(vals), 'b', markerfmt=" ", basefmt="-b")
    plt.stem(freq, np.abs(v2), 'r', markerfmt=" ", basefmt="-b") #graphing the max frequency in FFT for testing purposes
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')
    plt.xlim(0, sr/2) #Nyquist-Shannon and aliasing - disregard frequencies above sr/2
    plt.savefig('graphs/FFT/{}.png'.format(name))
    plt.show()
    # print(vals)

if __name__ == "__main__":
    mod_plots('sahil')
    mod_plots("ria")
    mod_plots("alex")
    mod_plots("anshul")
    print(max_freq)