from scipy.signal import filtfilt
import scipy

import matplotlib.pyplot as plt


def bandpass_filter(signal_arr, sr, gender = None):

    for ind, signal in enumerate(signal_arr):
        nyq_lim = sr / 2

        print("before")
        f, t, Sxx = scipy.signal.spectrogram(signal, sr)
        plt.pcolormesh(t, f, Sxx, shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()

        lowcut, highcut = [50,300] if gender is None else [125, 275] if gender == 'f' else [50,200]

        # lowcut = 50
        # highcut = 300

        low = lowcut/nyq_lim
        high = highcut/nyq_lim

        order = 5

        b, a = scipy.signal.butter(order, [low, high], 'bandpass', analog=False)
        filtered = scipy.signal.filtfilt(b, a, signal)

        # print("after")
        # librosa.display.waveshow(filtered)
        # plt.show()
        dist = 0
        for i in range(len(signal)):
            dist += (signal[i] - filtered[i])**2

        print("after")
        f, t, Sxx = scipy.signal.spectrogram(filtered, sr)
        plt.pcolormesh(t, f, Sxx, shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()

        print(f'Euc distance : {dist}')
        signal_arr[ind] = filtered

    return signal_arr
