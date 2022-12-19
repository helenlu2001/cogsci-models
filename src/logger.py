# very basic logging class
import numpy as np
class Logger():
    def __init__(self):
        self._log = {}
        self.factor = 4
    def log(self, key, value):
        if key not in self._log:
            self._log[key] = []
        self._log[key].append(value)
        print(key, ": ", value)
    def print(self):
        for k, v in self._log.items():
            print(k, ": ", np.mean(v) * self.factor)