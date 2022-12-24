from scipy.signal import filtfilt
import scipy


def bandpass_filter(signal_arr, sr):

    for ind, signal in enumerate(signal_arr):
        nyq_lim = sr / 2

        lowcut = 50
        highcut = 250

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

        print(f'Euc distance : {dist}')
        signal_arr[ind] = filtered

    return signal_arr
