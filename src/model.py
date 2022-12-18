# Defines the probabilistic models for the cogsci experiments

import numpy as np
import itertools as it

class ThreeArrowModel():
    # Implements:
    # Plan  <-  Execute
    #    \      /
    #     v    v
    #    Complete
    def __init__(self):
        self.variables = ['execute', 'plan', 'complete']
        self.dependencies = {
            'execute': [], 
            'plan': ['execute'], 
            'complete': ['execute', 'plan']
            }
        self.ps = {
            'execute': np.array(0.2),
            'plan': np.array([0.2, 0.3]),
            'complete': np.array([[0.2, 0.3], [0.4, 0.5]])
        }

    def nll(self, data, v, verbose=False):
        """ 
        Data is a dictionary with keys 'execute', 'plan', 'complete'
        Each value is an array of n observations, which are probabilities from 0 to 1
        v is the variable to compute the nll for
        """
        causers = [np.array((1-data[k], data[k])) for k in self.dependencies[v]]
        predicted = self.ps[v][..., None]
        for causer in causers:
            predicted = np.einsum('ij,i...j -> ...j', causer, predicted)
        if verbose:
            print(predicted, data[v])
        return self.nll_function(predicted, data[v])

    def nll_function(self, predicted, target):
        return -np.sum(target * np.log(predicted) + (1-target) * np.log(1-predicted))


