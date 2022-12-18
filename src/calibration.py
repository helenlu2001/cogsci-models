# Convert between time, probability and difficulty score

# time: 0 - infinity; + correlation
# prob: 0 - 1; - correlation
# diff: 1 - 10; + correlation

import numpy as np

class Converter():
    def __init__(self):
        self.halflife = 10 # minutes
    def time_to_prob(self, time):
        results = np.zeros_like(time, dtype=np.float)
        valid = (time.astype(str) != np.array(['not possible']))
        results[valid] = np.exp(-(time[valid].astype(np.float)) / self.halflife)
        return results
    def diff_to_prob(self, diff):
        return 1 - (diff-1) / 9 # 1-10 -> 0-1
    