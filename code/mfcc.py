from python_speech_features import mfcc

def calc_mfcc(processed_sigs):
    mfccs = []
    sr = 800
    for i in range(len(processed_sigs)):

        mfcc_arr = mfcc(processed_sigs[i], samplerate = sr, highfreq = sr/2, winstep = len(processed_sigs[i])/sr + 0.01, nfft = 256)
        # print(mfcc_arr.shape) #testing shape of returned MFCC array - why are there 24 coeffs. for some, and 12 (as desired) for others?
        mfccs.append(mfcc_arr[0].tolist())
    return mfccs

if __name__ == "__main__":
    mfccs = calc_mfcc('dan')
    # calc_mfcc('khachane')
    print('working')

