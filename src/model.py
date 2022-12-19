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

def planning_models():
    # Consideration model
    def consideration(model, x, dataset=None):
        if x[0] > x[1]:
            return np.inf
        model.ps['plan'] = x
        return model.nll(dataset, 'plan')
    
    # Independent model
    def independent(model, x, dataset=None):
        x = x.item()
        model.ps['plan'] = np.array([x, x])
        return model.nll(dataset, 'plan')
    
    return {
        'Consideration': (consideration, (2,)),
        'Independent': (independent, (1,))
    }

def completion_models():
    # Arbitary dependence model
    def no_prior(model, x, dataset=None):
        model.ps['complete'] = x
        return model.nll(dataset, 'complete')

    ## Human prior, 
    ## Here we assume that a low probability of completing the task if we didn't plan or didn't execute
    def human_prior(model, x, dataset=None):
        low, high = x
        if low > high:
            return np.inf
        model.ps['complete'] = np.array([[low, low], [low, high]]) # complete iff plan and execute
        return model.nll(dataset, 'complete')

    ## No-plan model
    ## Here we assume that completing the task is independent of completing the planning
    def no_plan(model, x, dataset=None):
        low, high = x
        if low > high:
            return np.inf
        model.ps['complete'] = np.array([[low, low], [high, high]])
        return model.nll(dataset, 'complete')

    return {
        'No-prior': (no_prior, (2,2)),
        'Intuitive': (human_prior, (2,)),
        'No-plan': (no_plan, (2,))
    }

