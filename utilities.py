import datetime
import matplotlib.pyplot as plt
import pandas as pd
from math import ceil, sqrt
import functools
import numpy as np
import collections
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os


################################################################################
#
# Part 1 : Some simple convenience functions
#
################################################################################

def compose(*funcs):
    outer = funcs[:-1][::-1] # Python only provides left folds
    def composed_func(*args, **kwargs):
        inner = funcs[-1](*args, **kwargs)
        return functools.reduce(lambda val, func : func(val), outer, inner)
    return composed_func

# Aliases for filters and maps
lfilter        = compose(list, filter)
lmap           = compose(list, map)
afilter        = compose(np.array, list, filter)
filternull     = functools.partial(filter, None)
lfilternull    = compose(list, filternull)
filternullmap  = compose(filternull, map)
lfilternullmap = compose(lfilternull, map)

def list_diff(a, b):
    return list(set(a).difference(b))

def print_dict(d):
    key_len = max(map(len, d.keys()))
    for k, v in d.items():
        print(f'{k.ljust(key_len)} : {v}')



################################################################################
#
# Part 2 : Some simple convenience functions for dropping data in dataframes
#
################################################################################

def drop_col(df, *cols):
    df.drop(columns = list(cols), inplace = True)


def drop_by_rule(df, bool_series):
    index = df[bool_series].index
    df.drop(index = index, inplace = True)

################################################################################
#
# Part 3 : Convenience functions for plotting
#
################################################################################

# With pyplot interactive mode off (via plt.ioff() call)
#  every call needs a figure and axis created
#  this provides a simple way to create such a figure
def plot(fn, *args, **kwargs):
    if 'figsize' in kwargs:
        fig, ax = plt.subplots(figsize = kwargs['figsize'])
        del kwargs['figsize']
    else:
        fig, ax = plt.subplots()
    kwargs['ax'] = ax
    fn(*args, **kwargs)
    return fig


################################################################################
#
# Part 4 : Timing
#
################################################################################

# Super simple timer
#  Timing implemented as class methods
#  to avoid having to instantiate
class Timer:

    @classmethod
    def start(cls):
        cls.start_time = datetime.datetime.now()

    @classmethod
    def end(cls):
        delta     = datetime.datetime.now() - cls.start_time
        sec       = delta.seconds
        ms        = delta.microseconds // 1000
        cls.time  = f'{sec}.{ms}'
        print(f'{sec}.{ms} seconds elapsed')


################################################################################
#
# Part 5 : Persistence and timing conveniences
#
################################################################################

from joblib import dump, load
fit_time_fname = './models/fit_times.joblib'

def _fit_time_interface(model_name, write = None):
    if os.path.exists(fit_time_fname):
        fit_time_dict = load(fit_time_fname)
    else:
        if not write:
            print('No fit time info found')
            return None
        fit_time_dict = {}

    if write:
        fit_time_dict[model_name] = write
        dump(fit_time_dict, fit_time_fname)
    else:
        if model_name in fit_time_dict:
            return fit_time_dict[model_name]
        else:
            print(f'No fit time found for {model_name}')
            return None

def get_fit_time(model_name):
    return _fit_time_interface(model_name)

def write_fit_time(model_name, fit_time):
    return _fit_time_interface(model_name, fit_time)

def get_time_per_fold(model_data, param_grid, splitter = None):
    fit_time = float(model_data['fit_time'])
    n_params = functools.reduce(lambda x, y : x * y,
        (len(v) for v in param_grid.values())
    )
    if splitter:
        try:
            n_folds = splitter.get_n_splits() * splitter.n_repeats
        except AttributeError:
            n_folds = splitter.get_n_splits()
    else:
        n_folds = 1

    return round(fit_time / (n_params * n_folds), 2)

################################################################################
#
# Part 6 : ROC Curve
#
################################################################################

def plot_roc_curve(fpr, tpr):
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label = 'ROC')
    xs = np.linspace(0, 1, len(fpr))
    ax.plot(xs, xs, label = 'Diagonal')
    ax.set_xlim([-0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_title('ROC Curve')
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.grid(True)
    ax.set_aspect(1)
    ax.legend()
    return fig
