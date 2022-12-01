import numpy as np
import librosa
import os

mfccs = dict()

def calc_mfcc(name):
    sr = 800
    num_at_name = len(os.listdir('recordings/{n}/'.format(n=name)))
    print(num_at_name)
    for i in range(1, num_at_name + 1):
        print('recordings/{n}/{n}{num}.m4a'.format(n=name, num=i))
        x, sr = librosa.load('recordings/{n}/{n}{num}.m4a'.format(n=name, num=i), sr=sr, mono = True)
        arr = np.where(x >= 0.01)  # thresholding the signal to find values that contain speech
        # librosa.display.waveshow(x[arr[0][0]:arr[0][-1]], sr = sr)

        x = x[arr[0][0]:arr[0][-1]]  # updating array to only include thresholded signal
        print(len(x))

        mfcc = librosa.feature.mfcc(y=x, sr = sr, n_mfcc = 12, n_fft = 256, htk = True)
        mfcc = np.reshape(mfcc, len(mfcc))
        print(mfcc)
        mfccs[name] = mfcc
        mfcc = []

if __name__ == "__main__":
    # calc_mfcc('sahil')
    # calc_mfcc("ria")
    # calc_mfcc("alex")
    # calc_mfcc("anshul")
    # calc_mfcc("dan")
    calc_mfcc('khachane')

    # alex = mfccs["alex"]
    # ria = mfccs["ria"]

    # import math
    # SE = 0
    # for i in range(12):
    #     SE += (alex[i]-ria[i])**2

    # mfcc_RMSE = math.sqrt(SE/12)
    # print("MFCC RMSE: {}".format(mfcc_RMSE))
