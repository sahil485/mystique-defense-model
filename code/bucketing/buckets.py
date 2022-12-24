import librosa
import matplotlib.pyplot as plt
import numpy as np

def calc_buckets(signal_arr):

    thresholded_sigs = []
    all_slopes = []

    for index, signal in enumerate(signal_arr):
        thresholded_sig, slopes = threshold(signal)
        thresholded_sigs.append(thresholded_sig)
        all_slopes.append(slopes)

    return [thresholded_sigs, all_slopes]


def threshold(signal):
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
                sub_sig_to_precede = signal[prev_sub_index_start : (1 + one_before) * len(signal) // buckets]
                max_portion = np.asarray(np.hstack((sub_sig_to_precede, max_portion)))
                break

    if prev_sub_index_start == -1:
        prev_sub_index_start = beg_max

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
                sub_sig_to_succeed = signal[one_after * len(signal) // buckets : post_sub_index_end]
                # sub_sig_to_succeed = signal[(3 + one_before) * len(signal) // buckets : post_sub_index_end]
                max_portion = np.asarray(np.hstack((max_portion, sub_sig_to_succeed)))

                break

    if post_sub_index_end == -1:
        post_sub_index_end = end_max

    librosa.display.waveshow(signal, sr=sr)
    plt.axvline(x=prev_sub_index_start / sr, color='r')
    plt.axvline(x=post_sub_index_end / sr, color='r')

    slopes = calc_linear_fit(max_portion, 5*buckets, prev_sub_index_start)

    plt.show()

    return [max_portion, slopes.tolist()]

#workflow - find two max buckets - store them
#for the two bordering buckets, divide into subbuckets and find those which have an average
    #greater than or equal to that of the max buckets - means signals are within them


def calc_linear_fit(max_portion, buckets, prev_sub_index_start):
    sr = 800
    slopes = np.zeros(buckets)

    for i in range(buckets):
        lower, upper = ((i * len(max_portion)) // buckets), (((i + 1) * len(max_portion)) // buckets) - 1

        curr_slope = (max_portion[upper] - max_portion[lower]) / (upper - lower)

        pt1 = [lower + prev_sub_index_start, max_portion[lower]]
        pt2 = [upper + prev_sub_index_start, max_portion[upper]]

        plt.plot([pt1[0] / sr, pt2[0] / sr], [pt1[1], pt2[1]], color='k')
        # plt.axvline(pt1[0] / sr, color='g')
        # plt.axvline(pt2[0] / sr, color='g')

        slopes[i] = curr_slope

    return slopes