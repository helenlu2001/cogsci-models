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
        return np.exp(-(self.time_fix(time)) / self.halflife * np.log(2))

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

def calibrate_plan_times():
    converter = Converter()
    converter.halflife = 10
    form_b = read_form_b()
    data = []
    for experiment in EXPERIMENTS:
        for toolset in ['main', 'other']:
            data.append((
                form_b[experiment]['plan']['prob'][toolset].values + 0.1,
                converter.time_to_prob(form_b[experiment]['plan']['time'][toolset].values)
            ))
    data = np.array(data)
    data = data.transpose(1,0,2).reshape(2, -1)
    plt.scatter(data[0], data[1])
    plt.savefig('plan_times.png')

if __name__ == '__main__':
    # calibrate_diff_to_time()
    calibrate_plan_times()
    