import numpy as np
from scipy.stats import kurtosis
import dill

def calc_kurtosis(processed_sigs):
    return [kurtosis(sig) for sig in processed_sigs]

if __name__ == "__main__":
    with open('../../pickles/dan.pkl', 'rb') as pkl:
        person = dill.load(pkl)
        print(calc_kurtosis(person.processed_sigs))