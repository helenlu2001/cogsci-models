from model import ThreeArrowModel
from calibration import Converter
from data import read_form_b, EXPERIMENTS
from logger import Logger

import numpy as np
import pandas as pd

def random_search(func, shape, n_iter=3000):
    """ 
    Minimize a function by randomly sampling
    """
    best = None
    best_x = None
    for _ in range(n_iter):
        x0 = np.array(np.random.rand(*shape))
        result = func(x0)
        if best is None or result < best:
            best = result
            best_x = x0
    return best_x

def calibrate(model, data, v):
    """
    Calibrate a model to data for a particular variable
    """
    def nll(x):
        model.ps[v] = x
        return model.nll(data, v)
    x0 = random_search(nll, model.ps[v].shape)
    model.ps[v] = x0
    return model

form_b = read_form_b()
# Convert the data to probabilities
converter = Converter()
plan_converter = Converter()
plan_converter.halflife = 5 # minutes, plan time is much faster

datasets = {}
for experiment in EXPERIMENTS:
    data = {
        'execute': converter.diff_to_prob(form_b[experiment]['execute']['difficulty']['main'].values),
        'plan': plan_converter.time_to_prob(form_b[experiment]['plan']['time']['main'].values),
        'complete': converter.time_to_prob(form_b[experiment]['complete']['time']['main'].values)
    }
    datasets[experiment] = pd.DataFrame(data)

def test_completion(logger, train, test):
    """
    Test the model predictions for completion
    """
    model = ThreeArrowModel()
    ## First two things are always calibrated
    calibrate(model, train, 'execute')
    calibrate(model, train, 'plan')

    # Calibrated model
    calibrate(model, train, 'complete')
    print('Calibrated parameters', model.ps)
    logger.log('Calibrated model NLL train', model.nll(train, 'complete'))
    logger.log('Calibrated model NLL test', model.nll(test, 'complete'))

    ## Human prior, 
    ## Here we assume that a low probability of completing the task if we didn't plan or didn't execute
    def human_prior(x):
        low, high = x
        if low > high:
            return np.inf
        model.ps['complete'] = np.array([[low, low], [low, high]]) # complete iff plan and execute
        return model.nll(train, 'complete')
    x0 = random_search(human_prior, (2,))
    model.ps['complete'] = np.array([[x0[0], x0[0]], [x0[0], x0[1]]])
    logger.log('Intuitive model NLL train', model.nll(train, 'complete'))
    logger.log('Intuitive model NLL test', model.nll(test, 'complete'))

    ## No-plan model
    ## Here we assume that completing the task is independent of completing the planning
    def no_plan(x):
        low, high = x
        if low > high:
            return np.inf
        model.ps['complete'] = np.array([[low, low], [high, high]])
        return model.nll(train, 'complete')
    x0 = random_search(no_plan, (2,))
    model.ps['complete'] = np.array([[x0[0], x0[0]], [x0[1], x0[1]]])
    logger.log('No-plan model NLL train', model.nll(train, 'complete'))
    logger.log('No-plan model NLL test', model.nll(test, 'complete'))

    ## Random model
    nlls_train = []
    nlls_test = []
    for _ in range(3000):
        model.ps['complete'] = np.random.rand(2,2)
        nlls_train.append(model.nll(train, 'complete'))
        nlls_test.append(model.nll(test, 'complete'))
    logger.log('Random model NLL train', np.mean(nlls_train))
    logger.log('Random model NLL test', np.mean(nlls_test))

def test_planning(logger, train, test):
    """
    Test model predictions for planning.
    """
    model = ThreeArrowModel()
    ## Execute is always calibrated
    calibrate(model, train, 'execute')

    # Calibrated model
    calibrate(model, train, 'plan')
    print('Calibrated parameters', model.ps)
    logger.log('Calibrated model NLL train', model.nll(train, 'plan'))
    logger.log('Calibrated model NLL test', model.nll(test, 'plan'))

    # Independent model
    def independent(x):
        x = x.item()
        model.ps['plan'] = np.array([x, x])
        return model.nll(train, 'plan')
    x0 = random_search(independent, (1,))
    model.ps['plan'] = np.array([x0.item(), x0.item()])
    logger.log('Independent model NLL train', model.nll(train, 'plan'))
    logger.log('Independent model NLL test', model.nll(test, 'plan'))

    # Random model
    nlls_train = []
    nlls_test = []
    for _ in range(3000):
        model.ps['plan'] = np.random.rand(2)
        nlls_train.append(model.nll(train, 'plan'))
        nlls_test.append(model.nll(test, 'plan'))
    logger.log('Random model NLL train', np.mean(nlls_train))
    logger.log('Random model NLL test', np.mean(nlls_test))

# Split the data into train and test
logger = Logger()
for test_experiment in EXPERIMENTS:
    print('Cross-validation for experiment', test_experiment)
    splits = {
        'train': [e for e in EXPERIMENTS if e != test_experiment],
        'test': [test_experiment]
    }
    train, test = pd.concat([datasets[e] for e in splits['train']]), pd.concat([datasets[e] for e in splits['test']])

    # Train and evaluate the model
    # test_completion(logger, train, test)
    test_planning(logger, train, test)

print('Overall results')
logger.print()