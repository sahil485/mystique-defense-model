import dill
import numpy as np


def calc_stddev(processed_sigs):
    return [np.std(sig) for sig in processed_sigs]

if __name__ == "__main__":
    with open('../../pickles/dan.pkl', 'rb') as pkl:
        person = dill.load(pkl)
        print(calc_stddev(person.processed_sigs))
    # print("hi")

