import numpy as np
from numpy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os

def calc_fft(processed_sigs):
    ffts = []
    sr = 800
    for i in range(len(processed_sigs)):

        sig = processed_sigs[i]

        # FFT PYTHON CALCULATION WORKFLOW:
        # DFT of signal calculated with FFT
        # the unit for the kth index is a frequency in terms of (k cycles/N samples) where N is the length of the signal
            # for example, the 100th value of fft(signal) represents the prominence of the frequency (100 cycles/ N samples)
            # or alternatively (100/N cycles/ 1 sample) in the sinusoidal construction of the signal
        # in order to convert (k cycles/N samples) into Hz, we multiply by the sampling rate (sr samples / 1 seconds)
            # we need to do this because we want to find the prominence of certain Hz values in the sinusoidal construction
            # of the signal because the unit (cycles/samples) is not of any importance to us
        # multiplying yields that the kth index of the FFT array represents the prominence of the frequency
            #(sr * k cycles / N seconds) Hz in the sinusoidal construction of the signal

        probabilities = fft(sig, 256) # take the FFT with 256 samples
        length = len(probabilities)

        # cycles_sample = np.arange(length) / length # construct the (cycles/sample) frequencies associated with each index of probabilities

        # AUTOMATED WITH np.fft.fftfreq

        freqs_Hz = fftfreq(length, d=1/sr)
        nyquist_limit = np.where(freqs_Hz < 0)[0][0] # last index at which the frequency is less than or equal to the Nyquist limit
                                                    # signal frequency distribution reflects about x = sr/2 after this point, pointless to graph/use
        # freqs_Hz = freqs_Hz[:nyquist_limit]
        probabilities = probabilities[:nyquist_limit]

        arr = probabilities.tolist()
        ffts.append(arr)

        # ind_max = np.argmax(np.abs(probabilities))  # find the position of the max probability in the FFT array
        # max_freq_Hz = (sr * ind_max) / length  # convert the frequency represented by that index from (cycles/sample) to (cycles/second)

        # OR YOU CAN DO IT WITH np.fft.fftfreq - both give the same answer as expected



        # plt.figure(figsize = (12, 6))
        # plt.subplot(121)
        #
        # plt.stem(freqs_Hz, np.abs(probabilities), 'r', markerfmt=" ", basefmt="-b") #graphing the FFT
        # plt.xlabel('Freq (Hz)')
        # plt.ylabel('FFT Amplitude |X(freq)|')
        # plt.xlim(0, sr/2) #Nyquist-Shannon and aliasing - disregard frequencies above sr/2
        # plt.savefig('graphs/FFT/{n}/{n}{num}.png'.format(n=name, num = i))
        # plt.show()
        # print("\n")

    # max_freqs["{n}".format(n=name)] = average
    # print("{n}'s Average Frequency: {m} Hz".format(n=name, m=average))

    return ffts

if __name__ == "__main__":
    # mod_plots('sahil')
    # mod_plots("ria")
    # mod_plots("alex")
    # mod_plots("anshul")
    # mod_plots("dan")
    calc_fft('khachane')