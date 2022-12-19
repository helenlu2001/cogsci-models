from model import ThreeArrowModel, planning_models, completion_models
from calibration import Converter
from data import read_form_b, EXPERIMENTS
from logger import Logger

import numpy as np
import pandas as pd
from functools import partial
from scipy.special import logsumexp

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
        'partial': converter.diff_to_prob(form_b[experiment]['partial']['difficulty']['main'].values),
        'execute': converter.diff_to_prob(form_b[experiment]['execute']['difficulty']['main'].values),
        'plan': plan_converter.time_to_prob(form_b[experiment]['plan']['time']['main'].values),
        'complete': converter.time_to_prob(form_b[experiment]['complete']['time']['main'].values)
        # 'plan': np.array(form_b[experiment]['plan']['prob']['main'].values),
        # 'complete': np.array(form_b[experiment]['complete']['prob']['main'].values)
    }
    datasets[experiment] = pd.DataFrame(data)

def cv_completion(logger, train, test):
    """
    Test the model predictions for completion
    """
    model = ThreeArrowModel()
    ## First two things are always calibrated
    calibrate(model, train, 'execute')
    calibrate(model, train, 'plan')
    for model_name, (model_fn, param_shape) in completion_models().items():
        x0 = random_search(partial(model_fn, model, dataset=train), param_shape)
        logger.log(f'{model_name} model NLL train', model_fn(model, x0, train))
        logger.log(f'{model_name} model NLL test',  model_fn(model, x0, test))

def evidence_completion(full):
    model = ThreeArrowModel()
    calibrate(model, full, 'execute')
    calibrate(model, full, 'plan')
    
    all_nlls = {}
    for model_name, (model_fn, param_shape) in completion_models().items():
        nlls = []
        for _ in range(10000):
            x0 = np.random.rand(*param_shape)
            n = model_fn(model, x0, full)
            if n != np.inf:
                nlls.append(n)
        nll = - logsumexp(-np.array(nlls)) + np.log(len(nlls))
        all_nlls[model_name] = nll
        print(f'{model_name} model NLL', nll)
    
    # softmax
    nlls = np.array(list(all_nlls.values()))
    base = - logsumexp(-nlls)
    for model_name, nll in all_nlls.items():
        print(f'{model_name} model relative likelihood', np.exp(base - nll))

def cv_planning(logger, train, test):
    """
    Test model predictions for planning.
    """
    model = ThreeArrowModel()
    ## Execute is always calibrated
    calibrate(model, train, 'execute')
    for model_name, (model_fn, param_shape) in planning_models().items():
        x0 = random_search(partial(model_fn, model, dataset=train), param_shape)
        logger.log(f'{model_name} model NLL train', model_fn(model, x0, train))
        logger.log(f'{model_name} model NLL test',  model_fn(model, x0, test))

def evidence_planning(full):
    """
    Evidence for planning models.
    """
    model = ThreeArrowModel()
    calibrate(model, full, 'execute')

    all_nlls = {}
    for model_name, (model_fn, param_shape) in planning_models().items():
        nlls = []
        for _ in range(10000):
            x0 = np.random.rand(*param_shape)
            n = model_fn(model, x0, full)
            if n != np.inf:
                nlls.append(n)
        nll = - logsumexp(-np.array(nlls)) + np.log(len(nlls))
        all_nlls[model_name] = nll
        print(f'{model_name} model NLL', nll)
    
    # softmax
    nlls = np.array(list(all_nlls.values()))
    base = - logsumexp(-nlls)
    for model_name, nll in all_nlls.items():
        print(f'{model_name} model relative likelihood', np.exp(base - nll))

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
    # cv_completion(logger, train, test)
    cv_planning(logger, train, test)

print('Final CV results')
logger.print()

full = pd.concat(datasets.values())
# evidence_completion(full)
# evidence_planning(full)


## Evidence (likelihood of data with random parameters)
