import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import numpy as np

def calc_mean(name, thres):
    sr = 800
    num_at_name = len(os.listdir('../recordings/{n}/'.format(n=name)))
    print(num_at_name)
    for i in range(1, num_at_name + 1):
        path = '../recordings/{n}/{n}{num}.m4a'.format(n=name, num=i)
        print(path)
        x, sr = librosa.load(path, sr=sr, mono=True)

        vals = np.where(x >= 0.02)
        print(vals[0][0]/sr)
        print(vals[0][-1]/sr)
        # print(x[vals[0][0] : vals[0][-1]])
        thres2 = np.average(x[vals[0][0] : vals[0][-1]])

        print(thres2)

        arr = np.where(x >= thres)  # thresholding the signal to find values that contain speech
        librosa.display.waveshow(x, sr=sr)
        plt.axhline(y = thres, color = 'r')
        plt.axvline(x = arr[0][0]/800, color = 'r')
        plt.axvline(x=arr[0][-1]/800, color='r')
        plt.title("Waveform: {}".format(path))
        plt.show()

        x = x[arr[0][0]:arr[0][-1]]  # updating array to only include thresholded signal
        librosa.display.waveshow(x, sr=sr)
        plt.axhline(y=thres, color='r')
        plt.axvline(x=x[0] / 800, color='r')
        plt.axvline(x=(arr[0][-1] - arr[0][0]) / 800, color='r')
        plt.show()

if __name__ == "__main__":
    # show_plots("sahil", 0.01)
    calc_mean("dan", 0.025)
    # calc_mean("khachane", 0.035)
    # calc_mean("ria", 0.035)