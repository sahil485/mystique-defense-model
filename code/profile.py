import pickle
import os
import librosa

from mfcc import calc_mfcc
from fft import calc_fft
from mean import calc_mean
from buckets import calc_buckets


class Profile:

    def __init__(self, name):
        self.name = name
        self.thres = 0.03
        self.sigs = self.get_signals()
        self.processed_sigs = self.calc_threshold() #will be different sizes because there are different numbers of thresholded signals
        self.mfccs = self.calc_mfccs()
        self.fft = self.calc_fft() #stored in complex ds; units are cycles/sample, not Hz -> shouldn't matter because they are all the same
        self.mean = self.calc_means()

    def calc_threshold(self):
        return calc_buckets(self.sigs)

    def calc_mfccs(self):
        return calc_mfcc(self.processed_sigs)

    def calc_fft(self):
        return calc_fft(self.processed_sigs)

    def calc_means(self):
        return calc_mean(self.processed_sigs)

    def get_signals(self): #retrieves signals from audio files
        original_signals = []
        sr = 800
        num_at_name = len(os.listdir(f'../recordings/{self.name}/'))
        for i in range(1, num_at_name + 1):
            path = f'../recordings/{self.name}/{self.name}{i}.m4a'
            x, _ = librosa.load(path, sr=sr, mono=True)
            original_signals.append(x)

        return original_signals

    def serialize(self):
        with open("../pickles/{}.pickle".format(self.name), 'wb') as outfile:
            pickle.dump(self, outfile)

    def getSigs(self): #returns the original signals
        return self.sigs

    def getThresholdedSigs(self): #returns the thresholded signals
        return self.processed_sigs

    def getMFCSS(self):
        return self.mfccs

    def getFFTs(self):
        return self.fft

    def getMean(self):
        return self.mean

if __name__ == "__main__":
    obj = Profile('ria')
    # print(obj.getMean())
    obj.serialize()