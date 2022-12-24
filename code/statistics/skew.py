from scipy.stats import skew
import dill


def calc_skew(processed_sigs):
    return [skew(sig) for sig in processed_sigs]


if __name__ == "__main__":
    with open('../../pickles/dan.pkl', 'rb') as pkl:
        person = dill.load(pkl)
        print(calc_skew(person.processed_sigs))