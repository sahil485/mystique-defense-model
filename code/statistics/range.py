import dill
import numpy as np

def calc_range(fft):
    print(len(fft))
    print(max(np.abs(fft[0])))
    print(fft)
    return

if __name__ == "__main__":
    with open('../../pickles/dan.pkl', 'rb') as pkl:
        f = dill.load(pkl)
        calc_range(f.fft)
