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
# Part 1 : Some simple convenience functions for maps and filters
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


class FeaturePlot:
    '''
        Manages a figure containing plots of many unrelated variables
        To use: this is an iterable that will yield (col_name, data, axis)
        for each variable it contains. For overlays, call overlay
    '''
    def __init__(self, *data, axsize = 4):
        self.data     = pd.concat(data, axis = 'columns')
        self.columns  = self.data.columns
        self.num_cols = len(self.columns)
        self._make_figure(axsize)

    def clone(self):
        return FeaturePlot(self.data)

    def _make_figure(self, axsize):
        '''
           Makes the main figure
        '''

        # Compute the size and get fig, axes
        s = ceil(sqrt(self.num_cols))
        fig, axes = plt.subplots(s, s, figsize = (axsize*s, axsize*s));
        axes = axes.ravel()

        # Delete excess axes
        to_delete = axes[self.num_cols:]
        for ax in to_delete:
            ax.remove()

        # Retain references
        self.fig  = fig
        self.axes = dict(zip(self.columns, axes))

        # Add titles
        for col, ax in self.axes.items():
            ax.set_title(col)

        self.grid_size = s

    def overlay(self, label, sharex = False, sharey = False):
        '''
            Adds a new layer of axes on top of an existing figure

            - Is a generator in similar style to self.__iter__ below.
            - A reference to the newly created axes is not maintained
                 by the class - the axes are intended to be single use.
                 If you want to access the axes later, either use the
                 matplotlib figure object or retain a reference
        '''
        for index, col in enumerate(self.columns):
            base_ax = self.axes[col]
            ax = self.fig.add_subplot(self.grid_size, self.grid_size, index + 1,
                                      sharex = base_ax if sharex else None,
                                      sharey = base_ax if sharey else None,
                                      label  = label,
                                      facecolor = 'none')

            for a in [ax, base_ax]:
                if not sharex:
                    a.tick_params(bottom = False,
                                  top = False,
                                  labelbottom = False,
                                  labeltop    = False)
                if not sharey:
                    a.tick_params(left = False,
                                  right = False,
                                  labelleft = False,
                                  labelright = False)


            yield col, self.data[col].values, ax

    def __iter__(self):
        for col in self.columns:
            yield col, self.data[col].values, self.axes[col]


################################################################################
#
# Part 4 : Miscellaneous conveniences
#
################################################################################

# Create a correlation matrix heat map
# Modified from the Seaborn Gallery
# https://seaborn.pydata.org/examples/many_pairwise_correlations.html
def correlation_matrix(data):
    # Compute the correlation matrix
    corr = data.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax = ax)

    return fig

# Create a table with the Variance Inflation Factors for each variable in a
#  dataframe of X-vars (exog in statsmodel speak)
# Essentially a wrapper for statsmodels variance_inflation_factor
def vif_table(df_x):
    return pd.Series({
        var_name : variance_inflation_factor(df_x.dropna().values, i)
        for i, var_name in enumerate(df_x.columns)
    }).sort_values(ascending = False)


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
# Part 5 : Class to keep track of models
#
################################################################################

class Model:
    '''
      A model is a collection of a y variable, a list of continuous
      variables, categorical variables, and interactions.
      Features:
       - can built patsy formulas for statsmodels formulas api to regression
       - can run OLS using regression
       - can save and restore a state

      Note: does not store data. Just remembers which IVs and DV are in a
       model
    '''

    def __init__(self, y, X_cat, X_cont, interactions = []):
        self.y = y
        self.X_cat = X_cat.copy()
        self.X_cont = X_cont.copy()
        self.interactions = interactions.copy()
        self.state_dict = {}
        self.save_state('init')

    def formula(self, include_cat = True, include_cont = True,
                      include_interactions = True, dv = None,
                      order = ['cat', 'cont', 'interactions']):
        if not dv:
            dv = self.y
        xvars = collections.defaultdict(lambda : [])
        if include_cat:
            xvars['cat']          = self._wrap_cat()
        if include_cont:
            xvars['cont']         = self.X_cont
        if include_interactions:
            xvars['interactions'] = self._wrap_interactions()

        xvars = sum(map(lambda type_ : xvars[type_], order), [])

        return dv + ' ~ ' + ' + '.join(xvars)

    def regress(self, data, **kwargs):
        results = smf.ols(self.formula(**kwargs), data).fit()
        return results

    def remove_var(self, *vars_):
        for var in vars_:
            success = False
            for list_ in [self.X_cat, self.X_cont, self.interactions]:
                try:
                    list_.remove(var)
                    success = True
                except ValueError:
                    pass
            if not success:
                print(f'Variable {var} not previously in the model')

    def add_cat_var(self, var):
        self.X_cat.append(var)

    def add_cont_var(self, var):
        self.X_cont.append(var)

    def add_interaction(self, *vars_, sep = '*'):
        if len(vars_) == 1:
            self.interactions.append(vars_[0])
        else:
            self.interactions.append(sep.join(vars_))

    def save_state(self, name):
        self.state_dict[name] = (self.y, self.X_cat.copy(), self.X_cont.copy(), self.interactions.copy())

    def restore_state(self, name):
        self.y, self.X_cat, self.X_cont, self.interactions = self.state_dict[name]
        self.X_cat, self.X_cont, self.interactions = \
            map(lambda list_ : list_.copy(),
                [self.X_cat, self.X_cont, self.interactions]
            )

    def _wrap_cat(self):
        return lmap(lambda var : f'C({var})', self.X_cat)

    def _wrap_interactions(self):
        return lmap(self._wrap_interaction, self.interactions)

    def _split_interaction(self, var):
        if '*' in var:
            split_on = '*'
        elif ':' in var:
            split_on = ':'
        else:
            raise ValueError(f'Not sure how to find the interaction in {var}')

        vars_ = var.split(split_on)
        return vars_, split_on

    def _wrap_interaction(self, var):
        vars_, split_on = self._split_interaction(var)
        vars_ = map(lambda var : f'C({var})' if var in self.X_cat else var, vars_)
        return split_on.join(vars_)

    def get_continuous_interactions(self):
        def is_cont_interaction(interaction):
            vars_, _ = self._split_interaction(interaction)
            for var in vars_:
                if var in self.X_cat:
                    return None
            return interaction
        return lfilternullmap(is_cont_interaction, self.interactions)


    def clone(self):
        out = Model(self.y, self.X_cat, self.X_cont, self.interactions)
        return out



################################################################################
#
# Part 6 : Convenience Functions for List Management
#
################################################################################

def list_diff(a, b):
    return list(set(a).difference(b))

def print_dict(d):
    key_len = max(map(len, d.keys()))
    for k, v in d.items():
        print(f'{k.ljust(key_len)} : {v}')

################################################################################
#
# Part 7 : Persistence and timing conveniences
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
# Part 8 : ROC Curve
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


################################################################################
#
# Part 9 : Inspection conveniences
#
################################################################################
