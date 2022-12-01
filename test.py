import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import os

def test(name):
    sr = 800
    num_at_name = len(os.listdir('recordings/{n}/'.format(n=name)))
    print(num_at_name)

    for i in range(1, num_at_name + 1):
        print('recordings/{n}/{n}{num}.m4a'.format(n=name, num=i))
        x, sr = librosa.load('recordings/{n}/{n}{num}.m4a'.format(n=name, num=i), sr=sr, mono = True)

        length = len(x)
        print(length)
        times = np.arange(length)/sr
        print(len(times))
        p1 = plt.figure(1)
        plt.stem(times, x, 'r', markerfmt=" ", basefmt="-b")
        # librosa.display.waveshow(x, sr=sr)
        # plt.title("Waveform: {}".format("Before"))


        # amplitude vs. time

        zeros = np.zeros(length)
        print(len(zeros))

        arr = np.where(np.abs(x) > 0.015)

        # print(arr[0])

        # print(arr[0])

        for ind in arr[0]:
            print(ind)
            zeros[ind] = x[ind]

        # x = x[arr[0][0]:arr[0][-1]]  # updating array to only include thresholded signal
        p2 = plt.figure(2)
        # librosa.display.waveshow(x[arr[0][0]:arr[0][-1]], sr=sr, color = 'r')
        plt.stem(times, zeros, 'b', markerfmt=" ", basefmt="-b")
        plt.title("Waveform: {}".format('AFTER'))

        plt.show()

        mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=12, n_fft=256, htk=True)
        mfcc = np.reshape(mfcc, len(mfcc))

if __name__ == "__main__":
    # test('sahil')
    # test("ria")
    # test("alex")
    # test("anshul")
    # test("dan")
    test('khachane')