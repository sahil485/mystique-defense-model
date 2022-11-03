import numpy as np
import librosa
import os


def calc_mfcc(name):
    sr = 800
    num_at_name = len(os.listdir('recordings/{n}/'.format(n=name)))
    print(num_at_name)
    for i in range(1, num_at_name + 1):
        mfcc = []
        x, sr = librosa.load('recordings/{n}/{n}{num}.m4a'.format(n=name, num=i), sr=sr)
        arr = np.where(x >= 0.01)  # thresholding the signal to find values that contain speech

        # librosa.display.waveshow(x[arr[0][0]:arr[0][-1]], sr = sr)
        # plt.title("Waveform: {}".format(name))
        # plt.savefig('graphs/waveform/{}_threshold.png'.format(name))

        x = x[arr[0][0]:arr[0][-1]]  # updating array to only include thresholded signal

        mfcc = librosa.feature.mfcc(y=x, sr = sr, n_mfcc = 12, n_fft = 256)
        mfcc = np.reshape(mfcc, len(mfcc))
        print(mfcc)

if __name__ == "__main__":
    calc_mfcc('sahil')
    calc_mfcc("ria")
    calc_mfcc("alex")
    calc_mfcc("anshul")
    # calc_mfcc("dan")