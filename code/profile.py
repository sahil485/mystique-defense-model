import dill
import os
import librosa

from bucketing.buckets import calc_buckets

from statistics.mfcc import calc_mfcc
from statistics.fft import calc_fft
from statistics.mean import calc_mean
from statistics.stddev import calc_stddev
from statistics.skew import calc_skew
from statistics.kurtosis import calc_kurtosis
from statistics.bandpass import bandpass_filter


class Profile:

    def __init__(self, name):
        self.name = name
        self.sr = 800
        self.sigs = self.get_signals()
        self.filtered = self.calc_bandpass()
        self.thresholded_sigs, self.slopes = self.calc_threshold() #will be different sizes because there are different numbers of thresholded signals
        self.mfccs = self.calc_mfccs()
        self.fft = self.calc_fft() #stored in complex ds; units are cycles/sample, not Hz -> shouldn't matter because they are all the same
        self.mean = self.calc_means()
        self.stddev = self.calc_stddev()
        self.skew = self.calc_skew()
        self.kurtosis = self.calc_kurtosis()

    def calc_threshold(self):
        return calc_buckets(self.sigs)

    def calc_bandpass(self):
        return bandpass_filter(self.sigs, sr = self.sr)

    def calc_mfccs(self):
        return calc_mfcc(self.thresholded_sigs)

    def calc_fft(self):
        return calc_fft(self.thresholded_sigs)

    def calc_means(self):
        return calc_mean(self.thresholded_sigs)

    def calc_stddev(self):
        return calc_stddev(self.thresholded_sigs)

    def calc_skew(self):
        return calc_skew(self.thresholded_sigs)

    def calc_kurtosis(self):
        return calc_kurtosis(self.thresholded_sigs)

    def get_signals(self): #retrieves signals from audio files
        original_signals = []
        sr = 800
        num_at_name = len(os.listdir(f'../recordings/{self.name}/'))
        for i in range(1, num_at_name + 1):
            # path = f'../recordings/{self.name}/{self.name}{i}.mp4'
            path = f'../recordings/{self.name}/{self.name}{i}.m4a'
            x, _ = librosa.load(path, sr=sr, mono=True)
            original_signals.append(x)

        return original_signals

    def serialize(self):
        with open("../pickles/{}.pkl".format(self.name), 'wb') as outfile:
            dill.dump(self, outfile)

    def getName(self):
        return self.name

    def getSigs(self): #returns the original signals
        return self.sigs

    def getThresholdedSigs(self): #returns the thresholded signals
        return self.thresholded_sigs

    def getSlopes(self):
        return self.slopes

    def getMFCSS(self):
        return self.mfccs

    def getFFTs(self):
        return self.fft

    def getMean(self):
        return self.mean

    def getSkew(self):
        return self.skew

    def getKurtosis(self):
        return self.kurtosis

if __name__ == "__main__":
    # for person in ['dan', 'khachane', 'ria', 'anshul', 'alex']:
    #     obj = Profile(person)
    #     obj.serialize()
    obj = Profile('dan')
