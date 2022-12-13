import pickle
from threshold import threshold
from mfcc import calc_mfcc
from fft import calc_fft

class Profile:

    def __init__(self, name):
        self.name = name
        self.thres = 0.03

        self.processed_sigs = self.threshold() #will be different sizes because there are different numbers of thresholded signals

        self.mfccs = self.calc_mfccs()

        self.fft = self.calc_fft() #stored in complex ds
                                    # units are cycles/sample, not Hz -> shouldn't matter because they are all the same

    def threshold(self):
        return threshold(self.name, self.thres)

    def calc_mfccs(self):
        return calc_mfcc(self.processed_sigs)

    def calc_fft(self):
        return calc_fft(self.processed_sigs)

    def serialize(self):
        with open("../pickles/{}.pickle".format(self.name), 'wb') as outfile:
            pickle.dump(self, outfile)

    def getProcessedSigs(self):
        return self.processed_sigs

    def getMFCSS(self):
        return self.mfccs

    def getFFTs(self):
        return self.fft

if __name__ == "__main__":
    obj = Profile('khachane')
    obj.serialize()