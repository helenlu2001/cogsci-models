# Convert between time, probability and difficulty score

# time: 0 - infinity; + correlation
# prob: 0 - 1; - correlation
# diff: 1 - 10; + correlation
from data import read_form_a, read_form_b, EXPERIMENTS

import matplotlib.pyplot as plt
import numpy as np

class Converter():
    def __init__(self):
        self.halflife = 60 # minutes
    def time_fix(self, time):
        results = np.zeros_like(time, dtype=float)
        valid = (time.astype(str) != np.array(['not possible']))
        results[valid] = time[valid].astype(float)
        results[~valid] = np.inf
        return results

    def time_to_prob(self, time):
        return np.exp(-(self.time_fix(time)) / self.halflife)

    def diff_to_prob(self, diff):
        return 1 - (diff-1) / 9 # 1-10 -> 0-1

def calibrate_diff_to_time():
    converter = Converter()
    form_a = read_form_a()
    form_b = read_form_b()
    data = []
    for experiment in EXPERIMENTS:
        for toolset in ['main', 'other']:
            print(form_a[experiment]['complete']['difficulty'][toolset])
            data.append((
                np.mean(form_a[experiment]['complete']['difficulty'][toolset].values),
                np.median(converter.time_fix(form_b[experiment]['complete']['time'][toolset].values))
            ))
    data = np.array(data)
    print(data)
    plt.scatter(data[:,0], data[:,1])
    plt.savefig('diff_to_time.png')

if __name__ == '__main__':
    calibrate_diff_to_time()
    