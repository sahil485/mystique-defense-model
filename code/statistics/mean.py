def calc_mean(processed_sig):
    amp_sum = 0
    num_samples = 0

    for i in range(len(processed_sig)):
        num_samples += len(processed_sig)
        for o in range(len(processed_sig[i])):
            amp_sum += abs(processed_sig[i][o])

    return amp_sum/num_samples
