import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import numpy as np


def threshold(name, thres):
    sigs = []
    sr = 800
    num_at_name = len(os.listdir('../recordings/{n}/'.format(n=name)))
    print(num_at_name)
    for i in range(1, num_at_name + 1):
        path = '../recordings/{n}/{n}{num}.m4a'.format(n=name, num=i)
        print(path)
        x, sr = librosa.load(path, sr=sr, mono=True)

        arr = np.where(x >= thres)  # thresholding the signal to find values that contain speech
        # librosa.display.waveshow(x, sr=sr)
        # plt.axhline(y = thres, color = 'r')
        # plt.axvline(x = arr[0][0]/800, color = 'r')
        # plt.axvline(x=arr[0][-1]/800, color='r')
        # plt.title("Waveform: {}".format(path))
        # plt.show()
        #
        # x = x[arr[0][0]:arr[0][-1]]  # updating array to only include thresholded signal
        # librosa.display.waveshow(x, sr=sr)
        # plt.axhline(y=thres, color='r')
        # plt.axvline(x=x[0] / 800, color='r')
        # plt.axvline(x=(arr[0][-1] - arr[0][0]) / 800, color='r')
        # plt.show()

        sigs.append(x)
    return sigs

if __name__ == "__main__":
    # show_plots("sahil", 0.01)
    threshold("dan", 0.025)
    # calc_mean("khachane", 0.035)
    # calc_mean("ria", 0.035)