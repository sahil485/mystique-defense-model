import librosa
import matplotlib.pyplot as plt
import numpy as np


def calc_buckets(signal_arr):

    thresholded_sigs = []

    for index, signal in enumerate(signal_arr):
        thresholded_sigs.append(calc_buckets_for(signal))

    return thresholded_sigs

def calc_buckets_for(signal):
    sr = 800
    buckets = 6
    averages = np.zeros(buckets)

    librosa.display.waveshow(signal, sr = 800)
    for i in range(buckets):
        lower, upper = i * len(signal) // buckets, (i+1) * len(signal) // buckets
        big_bucket_average = np.mean(np.abs(signal[lower : upper]))
        averages[i] = big_bucket_average
        plt.axvline(lower/sr)
        plt.axvline(upper/sr)

    print(averages)

    arr = np.argpartition(averages, (buckets-2, buckets-1))
    maxes = [arr[len(arr)-2], arr[len(arr)-1]]

    if maxes[0] > maxes[1]:
        maxes = [maxes[1], maxes[0]]

    beg_max, end_max = maxes[0] * len(signal)//buckets, (maxes[1] + 1) * len(signal) // buckets
    prev_sub_index_start, post_sub_index_end = beg_max, end_max
    max_portion = signal[beg_max : end_max]

    ##THRESHOLD PREVIOUS SUBBUCKET

    if maxes[0] > 0: # if a bucket exists before the first of the two largest (sequential)
        one_before = maxes[0]-1
        seg_before_first_max = signal[one_before * len(signal) // buckets : (1 + one_before) * len(signal) // buckets]
        librosa.display.waveshow(signal, sr=800)
        plt.show()

        for sub in range(buckets):
            librosa.display.waveshow(signal, sr=800)

            sub_lower_bound = ((one_before * len(signal)) + ((sub/buckets) * len(signal))) // buckets
            sub_upper_bound = ((one_before * len(signal)) + (((sub+1)/buckets) * len(signal))) // buckets
            sub_average = np.mean(np.abs(signal[int(sub_lower_bound) : int(sub_upper_bound)]))

            if sub_average >= (min(averages[maxes[0]], averages[maxes[1]]) * 0.35):
                prev_sub_index_start = int(sub_lower_bound)
                # print(f"Set lower bound {sub_lower_bound}")
                sub_sig_to_precede = signal[prev_sub_index_start : (1 + one_before) * len(signal) // buckets]
                max_portion = np.asarray(np.hstack((sub_sig_to_precede, max_portion)))

                break

    if prev_sub_index_start == -1:
        prev_sub_index_start = beg_max
        # print(f"Set beg max {beg_max}")

    ##THRESHOLD SUB BUCKET IMMEDIATELY AFTER

    if maxes[1] < buckets - 1: # if a bucket exists after the second of the two largest (sequential)
        one_after = maxes[1] + 1
        seg_after_second_max = signal[one_after * len(signal) // buckets: (one_after + 1) * len(signal) // buckets]

        for sub in range(buckets-1, -1, -1):
            librosa.display.waveshow(signal, sr=800)

            sub_lower_bound = ((one_after * len(signal)) + (((sub-1) / buckets) * len(signal))) // buckets
            sub_upper_bound = ((one_after * len(signal)) + ((sub / buckets) * len(signal))) // buckets
            sub_average = np.mean(np.abs(signal[int(sub_lower_bound): int(sub_upper_bound)]))

            if sub_average >= (min(averages[maxes[0]], averages[maxes[1]]) * 0.4):
                post_sub_index_end  = int(sub_upper_bound)
                # print(f"Set upper bound {sub_upper_bound}")
                sub_sig_to_succeed = signal[(1 + one_before) * len(signal) // buckets : post_sub_index_end]
                max_portion = np.asarray(np.hstack((max_portion, sub_sig_to_succeed)))

                break

    if post_sub_index_end == -1:
        post_sub_index_end  = end_max
        # print(f"Set end max {end_max}")

    librosa.display.waveshow(signal, sr=sr)
    plt.axvline(x=prev_sub_index_start / 800, color='g')
    plt.axvline(x=post_sub_index_end / 800, color='g')
    plt.show()

    return max_portion

#workflow - find two max buckets - store them
#for the two bordering buckets, divide into subbuckets and find those which have an average
    #greater than or equal to that of the max buckets - means signals are within them
