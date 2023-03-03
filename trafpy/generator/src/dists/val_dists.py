'''Module for generating value distributions.'''

from trafpy.generator.src.dists import plot_dists 
from trafpy.generator.src import tools

import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.inf)
from scipy.stats import skewnorm
from scipy import stats
import math
import matplotlib.pyplot as plt
import sys
import random

import ipywidgets as widgets
from ipywidgets import interact, interactive, interactive_output, interact_manual, fixed
from IPython.display import display, clear_output


def convert_key_occurrences_to_data(keys, num_occurrences):
    '''Converts value keys and their number of occurrences into random vars.

    Args:
        keys (list): Random variable values.
        num_occurrences (list): Number of each random variable to generate.

    Returns:
        list: Random variables generated.

    '''
    data = []
    idx = 0
    for key in keys:
        i = 0
        while i < num_occurrences[idx]:
            data.append(key)
            i += 1
        idx += 1

    return data


def convert_data_to_key_occurrences(data):
    '''Converts random variable data into value keys and corresponding occurrences.

    Args:
        data (list): Random variables to convert into key-num_occurrences pairs.

    Returns:
        dict: Random variable value - number of occurrences key-value pairs 
        generated from random variable data.
    
    '''
    # count number of times each var occurs in data
    counter_dict = {}
    for var in data:
        try:
            counter_dict[var] += 1
        except KeyError:
            # not yet encountered var val
            counter_dict[var] = 1

    return counter_dict

    

def x_round(x, round_to_nearest=1, num_decimal_places=2, print_data=False, min_val=None):
    '''Rounds variable to nearest specified value.'''

    if print_data:
        print('\nOriginal val: {}'.format(x))
        print(f'Round to nearest: {round_to_nearest}')
        print(f'Num decimal places: {num_decimal_places}')
        print(f'Min val: {min_val}')

    # factor = round(1/round_to_nearest, num_decimal_places)
    factor = 1/round_to_nearest

    if print_data:
        print(f'Factor: {factor}')

    rounded = round(round(x*factor)/factor, num_decimal_places)
    if min_val is not None:
        rounded = max(rounded, min_val)

    if print_data:
        print('Rounded val: {}'.format(rounded))

    return rounded
    

def gen_uniform_val_dist(min_val,
                         max_val,
                         round_to_nearest=None,
                         num_decimal_places=2,
                         occurrence_multiplier=100,
                         path_to_save=None,
                         plot_fig=False,
                         show_fig=False,
                         return_data=False,
                         xlim=None,
                         logscale=False,
                         rand_var_name='Random Variable',
                         prob_rand_var_less_than=None,
                         num_bins=0,
                         print_data=False):
    '''Generates a uniform distribution of random variable values.

    Uniform distributions are the most simple distribution. Each random variable
    value in a uniform distribution has an equal probability of occurring.

    Args:
        min_val (int/float): Minimum random variable value.
        max_val (int/float): Maximum random variable value.
        round_to_nearest (int/float): Value to round random variables to nearest.
            E.g. if round_to_nearest=0.2, will round each random variable to 
            nearest 0.2.
        num_decimal_places (int): Number of decimal places to random variable
            values. Need to explicitly state otherwise Python's floating point 
            arithmetic will cause spurious unique random variable value errors
            when discretising.
        occurrence_multiplier (int/float): When sampling random variables from
            distribution to create plot and random variable data, use this
            multiplier to determine number of data points to sample. A higher 
            value will cause the random variable data to match the probability
            distribution more closely, but will take longer to generate.
        path_to_save (str): Path to directory (with file name included) in which
            to save generated distribution. E.g. path_to_save='data/dists/my_dist'.
        plot_fig (bool): Whether or not to plot fig. If True, will return fig.
        show_fig (bool): Whether or not to plot and show fig. If True, will
            return and display fig.
        return_data (bool) Whether or not to return random variable data sampled
            from generated distribution.
        xlim (list): X-axis limits of plot. E.g. xlim=[0,10] to plot random
            variable values between 0 and 10.
        logscale (bool): Whether or not plot should have logscale x-axis and bins.
        rand_var_name (str): Name of random variable to label plot's x-axis.
        prob_rand_var_less_than (list): List of values for which to print the
            probability that a variable sampled randomly from the generated 
            distribution will be less than. This is useful for replicating 
            distributions from the literature. E.g. prob_rand_var_less_than=[3.7,5.8]
            will return the probability that a randomly chosen variable is less
            than 3.7 and 5.8 respectively.
        num_bins (int): Number of bins to use in plot. Default is 0, in which
            case the number of bins chosen will be automatically selected.
        print_data (bool): whether or not to print extra information about the
            generated data.

    Returns:
        tuple: Tuple containing:
            - **prob_dist** (*dict*): Probability distribution whose key-value pairs are 
              random variable value-probability pairs. 
            - **rand_vars** (*list, optional*): Random variable values sampled from the 
              generated probability distribution. To return, set return_data=True.
            - **fig** (*matplotlib.figure.Figure, optional*): Probability density 
              and cumulative distribution function plot. To return, set show_fig=True 
              and/or plot_fig=True.
    
    '''
    if round_to_nearest is None:
        # assume separation between vals is 1 unit
        separation = 1
    else:
        # separation between vals is same as round to nearest
        separation = round_to_nearest

    unique_vals = [val for val in np.arange(min_val, max_val+separation, separation)]
    if round_to_nearest is not None:
        unique_vals = [x_round(val,round_to_nearest,num_decimal_places,min_val=min_val) for val in unique_vals]
    else:
        # no need to discretise
        pass
    probabilities = [1/len(unique_vals) for _ in range(len(unique_vals))]

    prob_dist = {unique_val: prob for unique_val, prob in zip(unique_vals, probabilities)}
    
    if print_data:
        print('Prob dist:\n{}'.format(prob_dist))
    if path_to_save is not None:
        tools.pickle_data(path_to_save, prob_dist)
    if plot_fig or show_fig:
        min_prob = min(prob for prob in list(prob_dist.values()) if prob > 0)
        num_occurrences = [int(val*(1/min_prob)*occurrence_multiplier) for val in list(prob_dist.values())]
        rand_vars = convert_key_occurrences_to_data(unique_vals,num_occurrences)
        fig = plot_dists.plot_val_dist(rand_vars=rand_vars, 
                                       xlim=xlim,
                                       logscale=logscale,
                                       rand_var_name=rand_var_name,
                                       prob_rand_var_less_than=prob_rand_var_less_than,
                                       num_bins=num_bins,
                                       show_fig=show_fig,
                                       print_characteristics=False)
        if return_data:
            return prob_dist, rand_vars, fig
        else:
            return prob_dist, fig

    else:
        if return_data:
            return prob_dist, rand_vars
        else:
            return prob_dist
    
def gen_skew_dists(min_val,
                   max_val,
                   num_modes=2,
                   xlim=None,
                   rand_var_name='Unknown',
                   round_to_nearest=None,
                   num_decimal_places=2):
    
    # initialise widget and data dicts
    widget_dict = {mode_iter: {'num_bins_widget': widgets.IntText(description='bins:',value=0,step=1,disabled=False),
                               'num_samples_widget': widgets.BoundedIntText(description='Samples:',min=10,max=30000,value=10000,disabled=False),
                               'loc_widget': widgets.BoundedIntText(description='Location:',min=min_val,max=max_val,value=max_val/2,disabled=False),
                               'skew_widget': widgets.FloatText(description='Skew:',value=0.0,step=0.1,disabled=False),
                               'scale_widget': widgets.FloatText(description='Scale:',value=10.0,step=0.1,disabled=False,)}
                               for mode_iter in range(num_modes)}
    data_dict = {mode_iter: None for mode_iter in range(num_modes)}
    
    # customise skews
    locations = [None for i in range(num_modes)]
    skews = [None for i in range(num_modes)]
    scales = [None for i in range(num_modes)]
    num_skew_samples = [None for i in range(num_modes)]
    for mode_iter in range(num_modes):
        data_dict[mode_iter] = interactive(gen_skew_data, 
                                           {'manual': True},
                                           num_bins=widget_dict[mode_iter]['num_bins_widget'],
                                           num_skew_samples=widget_dict[mode_iter]['num_samples_widget'],
                                           location=widget_dict[mode_iter]['loc_widget'],
                                           skew=widget_dict[mode_iter]['skew_widget'],
                                           scale=widget_dict[mode_iter]['scale_widget'],
                                           min_val=fixed(min_val),
                                           max_val=fixed(max_val),
                                           xlim=fixed(xlim),
                                           rand_var_name=fixed(rand_var_name),
                                           round_to_nearest=fixed(round_to_nearest),
                                           num_decimal_places=fixed(num_decimal_places))

        
        # DISPLAY
        display(data_dict[mode_iter])
    
    return data_dict

def gen_skewnorm_val_dist(location, 
                  skew, 
                  scale, 
                  num_skew_samples=150000,
                  min_val=None,
                  max_val=None,
                  return_data=False,
                  xlim=None, 
                  plot_fig=False,
                  show_fig=False,
                  logscale=False,
                  path_to_save=None,
                  transparent=True,
                  rand_var_name='Random Variable',
                  num_bins=0,
                  round_to_nearest=None,
                  occurrence_multiplier=10,
                  prob_rand_var_less_than=None,
                  num_decimal_places=2):
    '''Generates a skew norm distribution of random variable values.

    Args:
        location (int/float): Position value of skewed distribution (mean shape
            parameter).
        skew (int/float): Skew value of skewed distribution (skewness shape
            parameter).
        scale (int/float): Scale value of skewed distribution (standard deviation
        scale (int/float): Scale value of skewed distribution (standard deviation
            shape parameter).
            shape parameter).
        num_skew_samples (int): Number of random variables to sample from distribution
            to generate skew data and plot.
        return_data (bool) Whether or not to return random variable data sampled
            from generated distribution.
        xlim (list): X-axis limits of plot. E.g. xlim=[0,10] to plot random
            variable values between 0 and 10.
        plot_fig (bool): Whether or not to plot fig. If True, will return fig.
        show_fig (bool): Whether or not to plot and show fig. If True, will
        logscale (bool): Whether or not plot should have logscale x-axis and bins.
        transparent (bool): Whether or not to make plot bins slightly transparent.
        rand_var_name (str): Name of random variable to label plot's x-axis.
        num_bins (int): Number of bins to use in plot. Default is 0, in which
            case the number of bins chosen will be automatically selected.
        round_to_nearest (int/float): Value to round random variables to nearest.
            E.g. if round_to_nearest=0.2, will round each random variable to 
            nearest 0.2.
        prob_rand_var_less_than (list): List of values for which to print the
            probability that a variable sampled randomly from the generated 
            distribution will be less than. This is useful for replicating 
            distributions from the literature. E.g. prob_rand_var_less_than=[3.7,5.8]
            will return the probability that a randomly chosen variable is less
            than 3.7 and 5.8 respectively.
        occurrence_multiplier (int/float): When sampling random variables from
            distribution to create plot and random variable data, use this
            multiplier to determine number of data points to sample. A higher 
            value will cause the random variable data to match the probability
            distribution more closely, but will take longer to generate.
        num_decimal_places (int): Number of decimal places to random variable
            values. Need to explicitly state otherwise Python's floating point 
            arithmetic will cause spurious unique random variable value errors
            when discretising.

    Returns:
        tuple: Tuple containing:
            - **prob_dist** (*dict*): Probability distribution whose key-value pairs are 
              random variable value-probability pairs. 
            - **rand_vars** (*list, optional*): Random variable values sampled from the 
              generated probability distribution. To return, set return_data=True.
            - **fig** (*matplotlib.figure.Figure, optional*): Probability density 
              and cumulative distribution function plot. To return, set show_fig=True 
              and/or plot_fig=True.

    '''
    rand_vars = []
    
    data = gen_skewnorm_data(a=skew,
                             loc=location,
                             scale=scale,
                             min_val=min_val,
                             max_val=max_val,
                             num_samples=num_skew_samples)
    rand_vars.append(list(data))
    
    rand_vars = [y for x in rand_vars for y in x] # flatten
    if round_to_nearest is not None:
        # discretise
        rand_vars = [x_round(i,round_to_nearest,num_decimal_places,min_val=min_val) for i in rand_vars]
    else:
        pass

    unique_vals, pmf = gen_discrete_prob_dist(rand_vars=rand_vars, 
                                              round_to_nearest=round_to_nearest,
                                              num_decimal_places=num_decimal_places)

    # ensure keys are floats so dont get error if try save with json
    prob_dist = {float(unique_val): prob for unique_val, prob in zip(unique_vals, pmf)}

    fig = None
    if path_to_save is not None:
        tools.pickle_data(path_to_save, prob_dist)
    if plot_fig or show_fig:
        min_prob = min(prob for prob in list(prob_dist.values()) if prob > 0)
        num_occurrences = [int(val*(1/min_prob)*occurrence_multiplier) for val in list(prob_dist.values())]
        rand_vars = convert_key_occurrences_to_data(unique_vals,num_occurrences)
        fig = plot_dists.plot_val_dist(rand_vars=rand_vars, 
                                       xlim=xlim,
                                       logscale=logscale,
                                       rand_var_name=rand_var_name,
                                       prob_rand_var_less_than=prob_rand_var_less_than,
                                       transparent=transparent,
                                       num_bins=num_bins,
                                       print_characteristics=False,
                                       show_fig=show_fig)

    return prob_dist, rand_vars, fig


def gen_skew_data(location, 
                  skew, 
                  scale, 
                  min_val,
                  max_val,
                  num_skew_samples,
                  xlim=None, 
                  logscale=False,
                  transparent=True,
                  rand_var_name='Unknown',
                  num_bins=0,
                  round_to_nearest=None,
                  num_decimal_places=2):
    '''Generates and plots skewed data for interactive multimodal distributions.

    Args:
        location (int/float): Position value of skewed distribution (mean shape
            parameter).
        skew (int/float): Skew value of skewed distribution (skewness shape
            parameter).
        scale (int/float): Scale value of skewed distribution (standard deviation
        scale (int/float): Scale value of skewed distribution (standard deviation
            shape parameter).
            shape parameter).
        num_skew_samples (int): Number of random variables to sample from distribution
            to generate skew data and plot.
        xlim (list): X-axis limits of plot. E.g. xlim=[0,10] to plot random
            variable values between 0 and 10.
        logscale (bool): Whether or not plot should have logscale x-axis and bins.
        transparent (bool): Whether or not to make plot bins slightly transparent.
        rand_var_name (str): Name of random variable to label plot's x-axis.
        num_bins (int): Number of bins to use in plot. Default is 0, in which
            case the number of bins chosen will be automatically selected.
        round_to_nearest (int/float): Value to round random variables to nearest.
            E.g. if round_to_nearest=0.2, will round each random variable to 
            nearest 0.2.
        num_decimal_places (int): Number of decimal places to random variable
            values. Need to explicitly state otherwise Python's floating point 
            arithmetic will cause spurious unique random variable value errors
            when discretising.

    Returns:
        list: Random variable values sampled from distribution.

    '''
    skew_data = []
    
    data = gen_skewnorm_data(a=skew,
                             loc=location,
                             scale=scale,
                             min_val=min_val,
                             max_val=max_val,
                             num_samples=num_skew_samples)
    skew_data.append(list(data))
    
    skew_data = [y for x in skew_data for y in x] # flatten
    if round_to_nearest is not None:
        # discretise
        skew_data = [x_round(i,round_to_nearest,num_decimal_places,min_val=min_val) for i in skew_data]
    else:
        pass

    data_description = stats.describe(skew_data) 
    print('Characteristics of generated distribution:\n{}'.format(data_description))
    
    plot_dists.plot_val_dist(skew_data, 
                             xlim=xlim, 
                             logscale=logscale,
                             num_bins=num_bins,
                             transparent=transparent,
                             rand_var_name=rand_var_name,
                             print_characteristics=False)

    return skew_data

def combine_multiple_mode_dists(data_dict,
                                min_val,
                                max_val,
                                xlim=None,
                                rand_var_name='Unknown',
                                round_to_nearest=None,
                                num_decimal_places=2):
    bg_factor = widgets.FloatText(min=0,max=10,step=0.001,value=0.5)
    num_bins_widget = widgets.IntText(description='bins:',value=0,step=1,disabled=False)
    
    prob_dist = interactive(combine_skews,
                           {'manual': True},
                           min_val=fixed(min_val),
                           max_val=fixed(max_val),
                           data_dict=fixed(data_dict), 
                           bg_factor=bg_factor,
                           xlim=fixed(xlim),
                           rand_var_name=fixed(rand_var_name),
                           num_bins=num_bins_widget,
                           round_to_nearest=fixed(round_to_nearest),
                           num_decimal_places=fixed(num_decimal_places))
    
    # DISPLAY
    display(prob_dist)

    return prob_dist

def combine_skews(data_dict, 
                  min_val,
                  max_val,
                  bg_factor=0.5,
                  xlim=None,
                  logscale=False,
                  transparent=True,
                  rand_var_name='Unknown',
                  num_bins=0,
                  round_to_nearest=None,
                  num_decimal_places=2):
    '''Combines multiple probability distributions for multimodal plotting.

    Args:
        data_dict (dict): Keys are mode iterations, values are random variable
            values for the mode iteration.
        min_val (int/float): Minimum random variable value.
        max_val (int/float): Maximum random variable value.
        bg_factor (int/float): Factor used to determine amount of noise to add
            amongst shaped modes being combined. Higher factor will add more
            noise to distribution and make modes more connected, lower will
            reduce noise but make nodes less connected.
        xlim (list): X-axis limits of plot. E.g. xlim=[0,10] to plot random
            variable values between 0 and 10.
        logscale (bool): Whether or not plot should have logscale x-axis and bins.
        rand_var_name (str): Name of random variable to label plot's x-axis.
        num_bins (int): Number of bins to use in plot. Default is 0, in which
            case the number of bins chosen will be automatically selected.
        round_to_nearest (int/float): Value to round random variables to nearest.
            E.g. if round_to_nearest=0.2, will round each random variable to 
            nearest 0.2.
        num_decimal_places (int): Number of decimal places to random variable
            values. Need to explicitly state otherwise Python's floating point 
            arithmetic will cause spurious unique random variable value errors
            when discretising.

    Returns:
        dict: Probability distribution whose key-value pairs are 
        random variable value-probability pairs. 

    '''
    locations = []
    skews = []
    scales = []
    num_skew_samples = []
    for mode_iter in range(len(data_dict)):
        print('Chosen skew {} stats: {}'.format(mode_iter+1,data_dict[mode_iter].kwargs))
        locations.append(data_dict[mode_iter].kwargs['location'])
        skews.append(data_dict[mode_iter].kwargs['skew'])
        scales.append(data_dict[mode_iter].kwargs['scale'])
        num_skew_samples.append(data_dict[mode_iter].kwargs['num_skew_samples'])
    
    prob_dist, data, fig = gen_multimodal_val_dist(min_val=min_val,
                                                  max_val=max_val,
                                                  locations=locations, 
                                                  skews=skews, 
                                                  scales=scales, 
                                                  bg_factor=bg_factor,
                                                  num_skew_samples=num_skew_samples,
                                                  round_to_nearest=round_to_nearest,
                                                  xlim=xlim,
                                                  logscale=logscale,
                                                  rand_var_name=rand_var_name,
                                                  return_data=True,
                                                  plot_fig=True,
                                                  num_bins=num_bins,
                                                  num_decimal_places=num_decimal_places)
    
    data_description = stats.describe(data) 
    print('Characteristics of generated distribution:\n{}'.format(data_description))
    
    return prob_dist


def gen_multimodal_val_dist(min_val,
                            max_val,
                            locations=[],
                            skews=[],
                            scales=[],
                            num_skew_samples=[],
                            bg_factor=0.5,
                            round_to_nearest=None,
                            num_decimal_places=2,
                            occurrence_multiplier=10,
                            path_to_save=None,
                            plot_fig=False,
                            show_fig=False,
                            return_data=False,
                            xlim=None,
                            logscale=False,
                            rand_var_name='Random Variable',
                            prob_rand_var_less_than=None,
                            num_bins=0,
                            print_data=False):
    '''Generates a multimodal distribution of random variable values.

    Multimodal distributions are arbitrary distributions with >= 2 different
    modes. A multimodal distribution with 2 modes is a special case called a 
    'bimodal distribution'. Bimodal distributions are the most common multi-
    modal distribution.

    E.g. Real-world scenarios of bimodal distributions: Starting salaries for
    lawyers, book prices, peak resaurant hours, age groups of disease victims,
    packet sizes in data centre networks, etc.

    Args:
        min_val (int/float): Minimum random variable value.
        max_val (int/float): Maximum random variable value.
        locations (list): Position value(s) of skewed distribution(s) (mean shape
            parameter).
        skews (list): Skew value(s) of skewed distribution(s) (skewness shape
            parameter).
        scales (list): Scale value(s) of skewed distribution(s) (standard deviation
            shape parameter).
        num_skew_samples (list): Number(s) of random variables to sample from distribution(s)
            to generate skew data and plot.
        bg_factor (int/float): Factor used to determine amount of noise to add
            amongst shaped modes being combined. Higher factor will add more
            noise to distribution and make modes more connected, lower will
            reduce noise but make nodes less connected.
        round_to_nearest (int/float): Value to round random variables to nearest.
            E.g. if round_to_nearest=0.2, will round each random variable to 
            nearest 0.2.
        num_decimal_places (int): Number of decimal places to random variable
            values. Need to explicitly state otherwise Python's floating point 
            arithmetic will cause spurious unique random variable value errors
            when discretising.
        occurrence_multiplier (int/float): When sampling random variables from
            distribution to create plot and random variable data, use this
            multiplier to determine number of data points to sample. A higher 
            value will cause the random variable data to match the probability
            distribution more closely, but will take longer to generate.
        path_to_save (str): Path to directory (with file name included) in which
            to save generated distribution. E.g. path_to_save='data/dists/my_dist'.
        plot_fig (bool): Whether or not to plot fig. If True, will return fig.
        show_fig (bool): Whether or not to plot and show fig. If True, will
        tuple
            return and display fig.
        return_data (bool) Whether or not to return random variable data sampled
            from generated distribution.
        xlim (list): X-axis limits of plot. E.g. xlim=[0,10] to plot random
            variable values between 0 and 10.
        logscale (bool): Whether or not plot should have logscale x-axis and bins.
        rand_var_name (str): Name of random variable to label plot's x-axis.
        prob_rand_var_less_than (list): List of values for which to print the
            probability that a variable sampled randomly from the generated 
            distribution will be less than. This is useful for replicating 
            distributions from the literature. E.g. prob_rand_var_less_than=[3.7,5.8]
            will return the probability that a randomly chosen variable is less
            than 3.7 and 5.8 respectively.
        num_bins (int): Number of bins to use in plot. Default is 0, in which
            case the number of bins chosen will be automatically selected.
        print_data (bool): Whether or not to print extra information about the
            generated data.

    Returns:
        tuple: Tuple containing:
            - **prob_dist** (*dict*): Probability distribution whose key-value pairs are 
              random variable value-probability pairs. 
            - **rand_vars** (*list, optional*): Random variable values sampled from the 
              generated probability distribution. To return, set return_data=True.
            - **fig** (*matplotlib.figure.Figure, optional*): Probability density 
              and cumulative distribution function plot. To return, set show_fig=True 
              and/or plot_fig=True.
    
    '''

    if round_to_nearest is None:
        # assume separation between vals is 1 unit
        separation = 1
    else:
        # separation between vals is same as round to nearest
        separation = round_to_nearest

    if len(num_skew_samples) == 0:
        # fill in automatically
        for _ in range(len(locations)):
            num_skew_samples.append(10000)

    num_bg_samples = int(sum(num_skew_samples)*bg_factor)
    num_modes = len(locations)

    poss_vals = np.arange(min_val,max_val+separation,separation)
    baseline_probs = np.ones((len(poss_vals)))/len(poss_vals)
    baseline_vals = list(np.random.choice(poss_vals,
                                          size=num_bg_samples,
                                          p=baseline_probs))

    skew_data = []
    skew_data.append(baseline_vals)
    for mode_iter in range(num_modes):
        data = gen_skewnorm_data(a=skews[mode_iter],
                                 loc=locations[mode_iter],
                                 scale=scales[mode_iter],
                                 min_val=min_val,
                                 max_val=max_val,
                                 num_samples=num_skew_samples[mode_iter])
        skew_data.append(list(data))

    skew_data = [y for x in skew_data for y in x] # flatten

    unique_vals, pmf = gen_discrete_prob_dist(rand_vars=skew_data, 
                                              round_to_nearest=round_to_nearest,
                                              num_decimal_places=num_decimal_places)
    
    # ensure keys are floats so dont get error if try save with json
    prob_dist = {float(unique_val): prob for unique_val, prob in zip(unique_vals, pmf)}

    if print_data:
        print('Prob dist:\n{}'.format(prob_dist))
    if path_to_save is not None:
        tools.pickle_data(path_to_save, prob_dist)
    if plot_fig or show_fig:
        min_prob = min(prob for prob in list(prob_dist.values()) if prob > 0)
        num_occurrences = [int(val*(1/min_prob)*occurrence_multiplier) for val in list(prob_dist.values())]
        rand_vars = convert_key_occurrences_to_data(unique_vals,num_occurrences)
        fig = plot_dists.plot_val_dist(rand_vars=rand_vars, 
                                       xlim=xlim,
                                       logscale=logscale,
                                       rand_var_name=rand_var_name,
                                       prob_rand_var_less_than=prob_rand_var_less_than,
                                       num_bins=num_bins,
                                       print_characteristics=False,
                                       show_fig=show_fig)
        if return_data:
            return prob_dist, rand_vars, fig
        else:
            return prob_dist, fig

    else:
        if return_data:
            return prob_dist, rand_vars 
        else:
            return prob_dist



def gen_skewnorm_data(a, 
                      loc, 
                      scale, 
                      num_samples,
                      min_val=None, 
                      max_val=None, 
                      round_to_nearest=None,
                      num_decimal_places=2,
                      interactive_params=None,
                      logscale=False,
                      transparent=True):
    '''Generates skew data.

    Args:
        a (int/float): Skewness shape parameter. When a=0, distribution is
            identical to a normal distribution.
        loc (int/float): Position value of skewed distribution (mean shape
            parameter).
        scale (int/float): Scale value of skewed distribution (standard deviation
            shape parameter).
        min_val (int/float): Minimum random variable value.
        max_val (int/float): Maximum random variable value.
        num_samples (int): Number of values to sample from generated distribution
            to generate skew data.

    Returns:
        list: List of random variable values sampled from skewed distribution.

    '''
    data = skewnorm(a, loc, scale).rvs(num_samples)
    if min_val is not None or max_val is not None:
        for data_iter in range(len(data)):
            counter = 0
            if min_val is not None and max_val is not None:
                while data[data_iter] < min_val or data[data_iter] > max_val:
                    data[data_iter] = skewnorm(a, loc, scale).rvs(size=1)
                    counter += 1
                    if counter > 10000:
                        sys.exit('scale too high for required max-min range')
            elif min_val is None and max_val is not None:
                while data[data_iter] > max_val:
                    data[data_iter] = skewnorm(a, loc, scale).rvs(size=1)
                    counter += 1
                    if counter > 10000:
                        sys.exit('scale too high for required max-min range')
            elif min_val is not None and max_val is None:
                while data[data_iter] < min_val:
                    data[data_iter] = skewnorm(a, loc, scale).rvs(size=1)
                    counter += 1
                    if counter > 10000:
                        sys.exit('scale too high for required max-min range')
            else:
                raise Exception('Bug')

    rand_vars = list(data.astype(float))

    if round_to_nearest is not None:
        rand_vars = [x_round(val, round_to_nearest, num_decimal_places, min_val=min_val) for val in rand_vars]

    if interactive_params is not None:
        # gen interactive plot
        data_description = stats.describe(rand_vars) 
        print('Characteristics of generated distribution:\n{}'.format(data_description))
        
        plot_dists.plot_val_dist(rand_vars, 
                                 xlim=interactive_params['xlim'], 
                                 logscale=logscale,
                                 num_bins=interactive_params['num_bins'],
                                 transparent=transparent,
                                 dist_fit_line=None,
                                 rand_var_name=interactive_params['rand_var_name'],
                                 print_characteristics=False,
                                 prob_rand_var_less_than=interactive_params['prob_rand_var_less_than'])

    return rand_vars


def gen_rand_vars_from_discretised_dist(unique_vars, 
                                        probabilities, 
                                        num_demands,
                                        jensen_shannon_distance_threshold=None,
                                        show_fig=False,
                                        xlabel='Random Variable',
                                        font_size=20,
                                        figsize=(4,3),
                                        marker_size=15,
                                        logscale=False,
                                        path_to_save=None):
    '''Generates random variable values by sampling from a discretised distribution.

    Args:
        unique_vars (list): Possible random variable values.
        probabilities (list): Corresponding probabilities of each random variable
            value being chosen.
        num_demands (int): Number of random variables to sample.
        jensen_shannon_distance_threshold (float): Maximum jensen shannon distance
            required of generated random variables w.r.t. discretised dist they're generated from.
            Must be between 0 and 1. Distance of 0 -> distributions are exactly the same.
            Distance of 1 -> distributions are not at all similar.
            https://medium.com/datalab-log/measuring-the-statistical-similarity-between-two-samples-using-jensen-shannon-and-kullback-leibler-8d05af514b15
            N.B. To meet threshold, this function will keep doubling num_demands
        show_fig (bool): Whether or not to generated sampled var dist plotted
            with the original distribution. 
        path_to_save (str): Path to directory (with file name included) in which
            to save generated data. E.g. path_to_save='data/my_data'

    Returns:
        numpy array: Random variable values sampled from dist.

    '''
    # random sampling
    if jensen_shannon_distance_threshold is not None:
        if jensen_shannon_distance_threshold <= 0 or jensen_shannon_distance_threshold > 1:
            raise Exception('jensen_shannon_distance_threshold must be >0 and <1, but is {}. See https://medium.com/datalab-log/measuring-the-statistical-similarity-between-two-samples-using-jensen-shannon-and-kullback-leibler-8d05af514b15'.format(jensen_shannon_distance_threshold))

        distance = 1
        num_demands_list = []
        distance_list = []
        max_list = []
        min_list = []
        mean_list = []
        std_list = []
        counter = 0
        while distance > jensen_shannon_distance_threshold:
            num_demands_list.append(num_demands)
            sampled_vars = np.random.choice(a=unique_vars, 
                                            size=num_demands,
                                            p=probabilities)
            # check similarity
            sampled_unique_vars, pmf = gen_discrete_prob_dist(sampled_vars, unique_vars=unique_vars)
            p, q = list(probabilities), list(pmf)
            distance = tools.compute_jensen_shannon_distance(p, q)
            distance_list.append(distance)
            max_list.append(max(sampled_vars))
            min_list.append(min(sampled_vars))
            mean_list.append(np.mean(sampled_vars))
            std_list.append(np.std(sampled_vars))
            if num_demands < 10:
                # increase to 10 or won't ever go beyond if increase by 10% each loop
                num_demands = 10
            else:
                num_demands = int(num_demands * 1.1)
            counter += 1
            if counter == 1e4:
                raise Exception('Looped 10,000 times and reached {} num_demands samples but distance {} still > threshold {}. Check no bugs, increase threshold and/or increase num_demands.'.format(num_demands, distance, jensen_shannon_distance_threshold))
    else:
        # no similarity threshold defined
        sampled_vars = np.random.choice(a=unique_vars, 
                                        size=num_demands,
                                        p=probabilities)
        sampled_unique_vars, pmf = gen_discrete_prob_dist(sampled_vars, 
                                                          unique_vars=unique_vars)

    if show_fig:
        print('Num demands needed for distance {}: {}'.format(jensen_shannon_distance_threshold, num_demands_list[-1]))

        # dist comparison
        plot_dict = {'Original': {'x_values': unique_vars, 'y_values': probabilities},
                     'Sampled': {'x_values': sampled_unique_vars, 'y_values': pmf}}
        _ = plot_dists.plot_val_scatter(plot_dict=plot_dict,
                                       xlabel=xlabel,
                                       ylabel='Probability',
                                       figsize=figsize,
                                       font_size=font_size,
                                       logscale=logscale,
                                       marker_style='+',
                                       alpha=[1, 0.6],
                                       marker_size=marker_size,
                                       use_scientific_notation_yaxis=True,
                                       use_scientific_notation_xaxis=True,
                                       show_fig=True)

        plot_dict = {'Original': {'x_values': unique_vars, 'y_values': probabilities}}
        _ = plot_dists.plot_val_scatter(plot_dict=plot_dict,
                                       xlabel=xlabel,
                                       ylabel='Probability',
                                       ghost_classes=['Sampled'],
                                       figsize=figsize,
                                       font_size=font_size,
                                       logscale=logscale,
                                       marker_style='+',
                                       alpha=1,
                                       marker_size=marker_size,
                                       use_scientific_notation_yaxis=True,
                                       use_scientific_notation_xaxis=True,
                                       show_fig=True)

        plot_dict = {'Sampled': {'x_values': sampled_unique_vars, 'y_values': pmf}}
        _ = plot_dists.plot_val_scatter(plot_dict=plot_dict,
                                       xlabel=xlabel,
                                       ylabel='Probability',
                                       ghost_classes=['Original'],
                                       figsize=figsize,
                                       font_size=font_size,
                                       logscale=logscale,
                                       marker_style='+',
                                       alpha=1,
                                       marker_size=marker_size,
                                       use_scientific_notation_yaxis=True,
                                       use_scientific_notation_xaxis=True,
                                       show_fig=True)

        # distance vs. num demands
        if jensen_shannon_distance_threshold is not None:
            plot_dict = {'sampled': {'x_values': num_demands_list, 'y_values': distance_list}}
            target = 0.1
            _ = plot_dists.plot_val_scatter(plot_dict=plot_dict,
                                           xlabel='Demands',
                                           ylabel='$\sqrt{JSD}$',
                                           xlim=[1e5, None],
                                           # ylim=[0,1],
                                           ylim=[0.05, 0.15],
                                           ghost_classes=['Sampled'],
                                           figsize=figsize,
                                           font_size=font_size,
                                           # logscale=logscale,
                                           logscale=False,
                                           horizontal_lines=[target],
                                           marker_style='+',
                                           plot_legend=False,
                                           alpha=1,
                                           # marker_size=marker_size,
                                           marker_size=100,
                                           use_scientific_notation_xaxis=True,
                                           show_fig=True)


            # _ = plt.figure()
            # plt.scatter(num_demands_list,
                        # min_list)
            # plt.xlabel('Number of Demands')
            # plt.ylabel('Min {}'.format(xlabel))
            # plt.show()
            plot_dict = {'sampled': {'x_values': num_demands_list, 'y_values': min_list}}
            target = min(unique_vars)
            _ = plot_dists.plot_val_scatter(plot_dict=plot_dict,
                                           xlabel='Demands',
                                           ylabel='Min {}'.format(xlabel),
                                           ghost_classes=['Sampled'],
                                           figsize=figsize,
                                           font_size=font_size,
                                           logscale=logscale,
                                           marker_style='+',
                                           horizontal_lines=[target],
                                           plot_legend=False,
                                           alpha=1,
                                           marker_size=marker_size,
                                           use_scientific_notation_yaxis=True,
                                           use_scientific_notation_xaxis=True,
                                           show_fig=True)

            # _ = plt.figure()
            # plt.scatter(num_demands_list,
                        # max_list)
            # plt.xlabel('Number of Demands')
            # plt.ylabel('Max {}'.format(xlabel))
            # plt.show()
            plot_dict = {'sampled': {'x_values': num_demands_list, 'y_values': max_list}}
            target = max(unique_vars)
            _ = plot_dists.plot_val_scatter(plot_dict=plot_dict,
                                           xlabel='Demands',
                                           ylabel='Max {}'.format(xlabel),
                                           ghost_classes=['Sampled'],
                                           figsize=figsize,
                                           font_size=font_size,
                                           logscale=logscale,
                                           marker_style='+',
                                           plot_legend=False,
                                           ylogscale=True,
                                           horizontal_lines=[target],
                                           alpha=1,
                                           marker_size=marker_size,
                                           use_scientific_notation_yaxis=True,
                                           use_scientific_notation_xaxis=True,
                                           show_fig=True)

            # _ = plt.figure()
            # plt.scatter(num_demands_list,
                        # mean_list)
            # plt.xlabel('Number of Demands')
            # plt.ylabel('Mean {}'.format(xlabel))
            # plt.show()
            plot_dict = {'sampled': {'x_values': num_demands_list, 'y_values': mean_list}}
            dist = stats.rv_discrete(values=(unique_vars, probabilities))
            target = dist.mean()
            _ = plot_dists.plot_val_scatter(plot_dict=plot_dict,
                                           xlabel='Demands',
                                           ylabel='Mean {}'.format(xlabel),
                                           ghost_classes=['Sampled'],
                                           figsize=figsize,
                                           font_size=font_size,
                                           logscale=logscale,
                                           marker_style='+',
                                           horizontal_lines=[target],
                                           plot_legend=False,
                                           alpha=1,
                                           marker_size=marker_size,
                                           use_scientific_notation_yaxis=True,
                                           use_scientific_notation_xaxis=True,
                                           show_fig=True)

            # _ = plt.figure()
            # plt.scatter(num_demands_list,
                        # std_list)
            # plt.xlabel('Number of Demands')
            # plt.ylabel('Std {}'.format(xlabel))
            # plt.show()
            plot_dict = {'sampled': {'x_values': num_demands_list, 'y_values': std_list}}
            dist = stats.rv_discrete(values=(unique_vars, probabilities))
            target = dist.std()
            _ = plot_dists.plot_val_scatter(plot_dict=plot_dict,
                                           xlabel='Demands',
                                           ylabel='Std {}'.format(xlabel),
                                           ghost_classes=['Sampled'],
                                           figsize=figsize,
                                           font_size=font_size,
                                           logscale=logscale,
                                           marker_style='+',
                                           plot_legend=False,
                                           horizontal_lines=[target],    
                                           alpha=1,
                                           marker_size=marker_size,
                                           use_scientific_notation_yaxis=True,
                                           use_scientific_notation_xaxis=True,
                                           show_fig=True)

    if path_to_save is not None:
        tools.pickle_data(path_to_save, sampled_vars)
    
    return sampled_vars


def gen_val_dist_data(val_dist, 
                      min_val, 
                      max_val, 
                      num_vals_to_gen,
                      path_to_save=None):
    '''Generates values between min_val and max_val following val_dist distribution'''
    raise Exception('CHRIS NOTE TO SELF 16/11/2020: Not sure what this function is for when have gen_rand_vars_from_discretised_dist() function. This funciton seems to be bugged and generates strange distribution random variables which differ from input distribution. At some point think should delete this function and replace all references to it with gen_rand_vars_from_discretised_dist(), but not sure atm where this function is being used. If this message pops up, comment out this Exception in val_dists.py and investiate bug or replace function.')
    array_sum = np.round(np.sum(val_dist),2)
    assert array_sum == 1, \
        'array must sum to 1, but is {}'.format(array_sum)
    
    poss_vals = np.arange(min_val, max_val+1)
    vals = np.zeros((num_vals_to_gen))
    
    # do multinomial exp to get number of each val
    counter_array = np.random.multinomial(num_vals_to_gen, 
                                          val_dist, 
                                          size=1)[0]

    # fill out vals array
    prev_val_iter = 0
    iter = np.nditer(np.asarray(poss_vals))
    for num_vals in list(counter_array):
        vals[prev_val_iter:prev_val_iter+num_vals] = next(iter)
        prev_val_iter += num_vals
    
    np.random.shuffle(vals) # randomly shuffle order
    
    if path_to_save is not None:
        tools.pickle_data(path_to_save, vals)
    
    return vals


def gen_discrete_prob_dist(rand_vars, 
                           unique_vars=None,
                           round_to_nearest=None, 
                           num_decimal_places=2,
                           path_to_save=None):
    '''Generate discrete probability distribution from list of random variables.

    Takes rand var values, rounds to nearest value (specified as arg, defaults
    by not rounding at all) to discretise the data, and generates a 
    probability distribution for the data

    Args:
        rand_vars (list): Random variable values
        unique_vars (list): List of unique random variables that can occur.
            If given, will init each as having occurred 0 times and count number
            of times each occurred. If left as None, will only record probabilities
            of random variables that actually occurred in rand_vars.
        round_to_nearest(int/float): Value to round rand vars to nearest when
            discretising rand var values. E.g. is round_to_nearest=0.2, will round each
            rand var to nearest 0.2
        num_decimal_places (int): Number of decimal places for discretised rand vars.
            Need to explitly state this because otherwise Python's floating point 
            arithmetic will cause spurious unique random var values

    Returns:
        tuple: Tuple containing:
            - **xk** (*list*): List of (discretised) unique random variable values 
              that occurred
            - **pmf** (*list*): List of corresponding probabilities that each
              unique value in xk occurs

    '''
    if round_to_nearest is not None:
        # discretise vars
        min_val = min(rand_vars)
        discretised_rand_vars = [x_round(rand_var,round_to_nearest,num_decimal_places,print_data=False,min_val=min_val) for rand_var in rand_vars]  
    else:
        # no further discretisation required
        discretised_rand_vars = rand_vars

    # count number of times each var occurs in data
    if unique_vars is None:
        # not given what unique rand vars are, init empty dict
        counter_dict = {}
    else:
        # given unique rand vars, init each one as having occurred 0 times
        counter_dict = {unique_var: 0 for unique_var in unique_vars}
    for var in discretised_rand_vars:
        try:
            counter_dict[var] += 1
        except KeyError:
            # not yet encountered var val
            counter_dict[var] = 1

    # convert counts to probabilities for each unique var
    total_num_vars = len(rand_vars)
    for unique_var in list(counter_dict.keys()):
        counter_dict[unique_var] /= total_num_vars

    # ensure all probabilities sum to 1
    total_sum = round(sum(list(counter_dict.values())),2)
    assert total_sum == 1, \
            'probabilities must sum to 1, but is {}'.format(total_sum)

    # gen probability distribution
    prob_dist = stats.rv_discrete(values=(list(counter_dict.keys()), list(counter_dict.values())))
    xk = list(counter_dict.keys())
    pmf = list(counter_dict.values())
    
    if path_to_save is not None:
        tools.pickle_data(path_to_save, prob_dist)

    return xk, pmf


def gen_exponential_dist(_beta, 
                         size,
                         round_to_nearest=None,
                         num_decimal_places=2,
                         min_val=None,
                         max_val=None,
                         interactive_params=None,
                         logscale=False,
                         transparent=True):
    '''Generates an exponential distribution of random variable values.
    
    The exponential distribution often fits scenarios whose events' random 
    variable values (e.g. 'time between events') are made of many small 
    values (e.g. time intervals) and a few large values. Often used to predict
    time until next event occurs.

    E.g. Real-world scenarios: Time between earthquakes, car accidents, mail
    delivery, and data centre demand arrival.

    Args:
        _beta (int/float): Mean random variable value.
        size (int): Number of random variable values to sample from distribution.
        interactive_params (dict): Dictionary of distribution parameter values
            (must provide if in interactive mode).
        logscale (bool): Whether or not plot should have logscale x-axis and bins.
        transparent (bool): Whether or not to make plot bins slightly transparent.

    Returns:
        list: Random variable values generated by sampling from the distribution.

    '''
    rand_vars = np.random.exponential(_beta,size=size)

    # check min and max vals 
    min_rand_var, max_rand_var = min(rand_vars), max(rand_vars)
    counter = 0
    if min_val is None and max_val is None:
        # don't need to worry about any min or max allowed values
        pass
    elif min_val is not None and max_val is None:
        # ensure greater than min_val
        while min(rand_vars) < min_val:
            min_idx = np.argmin(np.array(rand_vars))
            rand_vars[min_idx] = np.random.exponential(_beta,size=1)
            counter += 1
            if counter > 10000:
                sys.exit('Dist to broad for required min-max range. Increase min-max range or reduce dist broadness.')
    elif min_val is None and max_val is not None:
        # ensure less than max val
        while max(rand_vars) > max_val:
            max_idx = np.argmax(np.array(rand_vars))
            rand_vars[max_idx] = np.random.exponential(_beta,size=1)
            counter += 1
            if counter > 10000:
                sys.exit('Dist to broad for required min-max range. Increase min-max range or reduce dist broadness.')
    elif min_val is not None and max_val is not None:
        # ensure less than max val and greater than min val
        while min(rand_vars) < min_val or max(rand_vars) > max_val:
            if min(rand_vars) < min_val:
                min_idx = np.argmin(np.array(rand_vars))
                rand_vars[min_idx] = np.random.exponential(_beta,size=1)
            if max(rand_vars) > max_val:
                max_idx = np.argmax(np.array(rand_vars))
                rand_vars[max_idx] = np.random.exponential(_beta,size=1)
            counter += 1
            if counter > 10000:
                sys.exit('Dist to broad for required min-max range. Increase min-max range or reduce dist broadness.')

    if round_to_nearest is not None:
        rand_vars = [x_round(val, round_to_nearest, num_decimal_places, min_val=min_val) for val in rand_vars]
    else:
        # no need to discretise
        pass

    if interactive_params is not None:
        # gen interactive plot
        data_description = stats.describe(rand_vars) 
        print('Characteristics of generated distribution:\n{}'.format(data_description))
        
        plot_dists.plot_val_dist(rand_vars, 
                                 xlim=interactive_params['xlim'], 
                                 logscale=logscale,
                                 num_bins=interactive_params['num_bins'],
                                 transparent=transparent,
                                 dist_fit_line='exponential', 
                                 rand_var_name=interactive_params['rand_var_name'],
                                 print_characteristics=False,
                                 prob_rand_var_less_than=interactive_params['prob_rand_var_less_than'])

    return rand_vars


def gen_lognormal_dist(_mu, 
                       _sigma, 
                       size, 
                       round_to_nearest=None,
                       num_decimal_places=2,
                       min_val=None,
                       max_val=None,
                       interactive_params=None, 
                       logscale=False, 
                       transparent=True):
    '''Generates a log-normal distribution of random variable values.

    Log-normal distributions often fit scenarios whose random variable values
    have a low mean value but a high degree of variance, leading to a distribution
    that is positively skewed (i.e. has a long tail to the right of its peak).

    The log-normal distribution is mathematically similar to the normal distribution,
    since its random variable is notmally distributed when its logarithm is taken. 
    I.e. for a log-normally distributed random variable X, Y=ln(X) would have a
    normal distribution.

    E.g. of real-world scenarios: Length of a chess game, number of hospitalisations
    during an epidemic, the time after which a mechanical system needs repair, 
    data centre demand interarrival times, etc.

    Args:
        _mu (int/float): Mean value of underlying normal distribution.
        _sigma (int/float): Standard deviation of underlying normal distribution.
        size (int): Number of random variable values to sample from distribution.
        interactive_params (dict): Dictionary of distribution parameter values
            (must provide if in interactive mode).
        logscale (bool): Whether or not plot should have logscale x-axis and bins.
        transparent (bool): Whether or not to make plot bins slightly transparent.

    Returns:
        list: Random variable values generated by sampling from the distribution.

    '''
    rand_vars = stats.lognorm.rvs(s=_sigma, scale=math.exp(_mu), size=size)

    # check min and max vals 
    min_rand_var, max_rand_var = min(rand_vars), max(rand_vars)
    counter = 0
    if min_val is None and max_val is None:
        # don't need to worry about any min or max allowed values
        pass
    elif min_val is not None and max_val is None:
        # ensure greater than min_val
        while min(rand_vars) < min_val:
            min_idx = np.argmin(np.array(rand_vars))
            rand_vars[min_idx] = stats.lognorm.rvs(s=_sigma,scale=math.exp(_mu),size=1)
            counter += 1
            if counter > 10000:
                sys.exit('Dist to broad for required min-max range. Increase min-max range or reduce dist broadness.')
    elif min_val is None and max_val is not None:
        # ensure less than max val
        while max(rand_vars) > max_val:
            max_idx = np.argmax(np.array(rand_vars))
            rand_vars[max_idx] = stats.lognorm.rvs(s=_sigma,scale=math.exp(_mu),size=1)
            counter += 1
            if counter > 10000:
                sys.exit('Dist to broad for required min-max range. Increase min-max range or reduce dist broadness.')
    elif min_val is not None and max_val is not None:
        # ensure less than max val and greater than min val
        while min(rand_vars) < min_val or max(rand_vars) > max_val:
            if min(rand_vars) < min_val:
                min_idx = np.argmin(np.array(rand_vars))
                rand_vars[min_idx] = stats.lognorm.rvs(s=_sigma,scale=math.exp(_mu),size=1)
            if max(rand_vars) > max_val:
                max_idx = np.argmax(np.array(rand_vars))
                rand_vars[max_idx] = stats.lognorm.rvs(s=_sigma,scale=math.exp(_mu),size=1)
            counter += 1
            if counter > 10000:
                sys.exit('Dist to broad for required min-max range. Increase min-max range or reduce dist broadness.')

    if round_to_nearest is not None:
        rand_vars = [x_round(val, round_to_nearest, num_decimal_places, min_val=min_val) for val in rand_vars]
    else:
        # no need to discretise
        pass

    if interactive_params is not None:
        # gen interactive plot
        data_description = stats.describe(rand_vars) 
        print('Characteristics of generated distribution:\n{}'.format(data_description))
        
        plot_dists.plot_val_dist(rand_vars, 
                                 xlim=interactive_params['xlim'], 
                                 logscale=logscale,
                                 num_bins=interactive_params['num_bins'],
                                 transparent=transparent,
                                 dist_fit_line='lognormal',
                                 rand_var_name=interactive_params['rand_var_name'],
                                 print_characteristics=False,
                                 prob_rand_var_less_than=interactive_params['prob_rand_var_less_than'])
    return rand_vars

def gen_normal_dist(loc,
                    scale,
                    size, 
                    round_to_nearest=None,
                    num_decimal_places=2,
                    min_val=None,
                    max_val=None,
                    interactive_params=None,
                    logscale=False,
                    transparent=True):
    '''Generates a normal/gaussian distribution of random variable values.

    Args:
        size (int): number of random variable values to sample from distribution.
        interactive_params (dict): Dictionary of distribution parameter values
            (must provide if in interactive mode).
        logscale (bool): Whether or not plot should have logscale x-axis and bins.
        transparent (bool): Whether or not to make plot bins slightly transparent.
    
    Returns:
        list: random variable values generated by sampling from the distribution.

    '''
    rand_vars = np.random.normal(loc=loc, scale=scale, size=size)

    # check min and max vals 
    min_rand_var, max_rand_var = min(rand_vars), max(rand_vars)
    counter = 0
    if min_val is None and max_val is None:
        # don't need to worry about any min or max allowed values
        pass
    elif min_val is not None and max_val is None:
        # ensure greater than min_val
        while min(rand_vars) < min_val:
            min_idx = np.argmin(np.array(rand_vars))
            rand_vars[min_idx] = np.random.normal(loc=loc, scale=scale, size=1)
            counter += 1
            if counter > 10000:
                sys.exit('Dist to broad for required min-max range. Increase min-max range or reduce dist broadness.')
    elif min_val is None and max_val is not None:
        # ensure less than max val
        while max(rand_vars) > max_val:
            max_idx = np.argmax(np.array(rand_vars))
            rand_vars[max_idx] = np.random.normal(loc=loc, scale=scale, size=1)
            counter += 1
            if counter > 10000:
                sys.exit('Dist to broad for required min-max range. Increase min-max range or reduce dist broadness.')
    elif min_val is not None and max_val is not None:
        # ensure less than max val and greater than min val
        while min(rand_vars) < min_val or max(rand_vars) > max_val:
            if min(rand_vars) < min_val:
                min_idx = np.argmin(np.array(rand_vars))
                rand_vars[min_idx] = np.random.normal(loc=loc, scale=scale, size=1)
            if max(rand_vars) > max_val:
                max_idx = np.argmax(np.array(rand_vars))
                rand_vars[max_idx] = np.random.normal(loc=loc, scale=scale, size=1)
            counter += 1
            if counter > 10000:
                sys.exit('Dist to broad for required min-max range. Increase min-max range or reduce dist broadness.')

    if round_to_nearest is not None:
        rand_vars = [x_round(val, round_to_nearest, num_decimal_places, min_val=min_val) for val in rand_vars]
    else:
        # no need to discretise
        pass
    
    if interactive_params is not None:
        # gen interactive plot
        data_description = stats.describe(rand_vars) 
        print('Characteristics of generated distribution:\n{}'.format(data_description))
        
        plot_dists.plot_val_dist(rand_vars, 
                                 xlim=interactive_params['xlim'], 
                                 logscale=logscale,
                                 num_bins=interactive_params['num_bins'],
                                 transparent=transparent,
                                 dist_fit_line='normal', 
                                 rand_var_name=interactive_params['rand_var_name'],
                                 print_characteristics=False,
                                 prob_rand_var_less_than=interactive_params['prob_rand_var_less_than'])
    return rand_vars


def gen_pareto_dist(_alpha, 
                    _mode,  
                    size, 
                    round_to_nearest=None,
                    num_decimal_places=2,
                    min_val=None,
                    max_val=None,
                    interactive_params=None,
                    logscale=False,
                    transparent=True):
    '''Generates a pareto distribution of random variable values.

    Pareto distributions often fit scenarios whose random variable values
    have high probability of having a small range of values, leading to a 
    distribution that is heavily skewed (i.e. has a long tail).

    E.g. real-world scenarios: A large portion of society's wealth being held
    by a small portion of its population, human settlement sizes, value of
    oil reserves in oil fields, size of sand particles, male dating success
    on Tinder, sizes of data centre demands, etc.

    Args:
        _alpha (int/float): Shape parameter of Pareto distribution. Describes
            how 'stretched out' (i.e. how high variance) the distribution is.
        _mode (int/float): Mode of the distribution, which is also the distribution's
            minimum possible value. 
        size (int): number of random variable values to sample from distribution.
        interactive_params (dict): Dictionary of distribution parameter values
            (must provide if in interactive mode).
        logscale (bool): Whether or not plot should have logscale x-axis and bins.
        transparent (bool): Whether or not to make plot bins slightly transparent.
    
    Returns:
        list: random variable values generated by sampling from the distribution.

    '''
    rand_vars = stats.pareto.rvs(b=_alpha, loc=0, scale=_mode, size=size)

    # check min and max vals 
    min_rand_var, max_rand_var = min(rand_vars), max(rand_vars)
    counter = 0
    if min_val is None and max_val is None:
        # don't need to worry about any min or max allowed values
        pass
    elif min_val is not None and max_val is None:
        # ensure greater than min_val
        while min(rand_vars) < min_val:
            min_idx = np.argmin(np.array(rand_vars))
            rand_vars[min_idx] = stats.pareto.rvs(b=_alpha, loc=0, scale=_mode, size=1)
            counter += 1
            if counter > 10000:
                sys.exit('Dist to broad for required min-max range. Increase min-max range or reduce dist broadness.')
    elif min_val is None and max_val is not None:
        # ensure less than max val
        while max(rand_vars) > max_val:
            max_idx = np.argmax(np.array(rand_vars))
            rand_vars[max_idx] = stats.pareto.rvs(b=_alpha, loc=0, scale=_mode, size=1)
            counter += 1
            if counter > 10000:
                sys.exit('Dist to broad for required min-max range. Increase min-max range or reduce dist broadness.')
    elif min_val is not None and max_val is not None:
        # ensure less than max val and greater than min val
        while min(rand_vars) < min_val or max(rand_vars) > max_val:
            if min(rand_vars) < min_val:
                min_idx = np.argmin(np.array(rand_vars))
                rand_vars[min_idx] = stats.pareto.rvs(b=_alpha, loc=0, scale=_mode, size=1)
            if max(rand_vars) > max_val:
                max_idx = np.argmax(np.array(rand_vars))
                rand_vars[max_idx] = stats.pareto.rvs(b=_alpha, loc=0, scale=_mode, size=1)
            counter += 1
            if counter > 10000:
                sys.exit('Dist to broad for required min-max range. Increase min-max range or reduce dist broadness.')

    if round_to_nearest is not None:
        rand_vars = [x_round(val, round_to_nearest, num_decimal_places, min_val=min_val) for val in rand_vars]
    else:
        # no need to discretise
        pass
    
    if interactive_params is not None:
        # gen interactive plot
        data_description = stats.describe(rand_vars) 
        print('Characteristics of generated distribution:\n{}'.format(data_description))
        
        plot_dists.plot_val_dist(rand_vars, 
                                 xlim=interactive_params['xlim'], 
                                 logscale=logscale,
                                 num_bins=interactive_params['num_bins'],
                                 transparent=transparent,
                                 dist_fit_line='pareto', 
                                 rand_var_name=interactive_params['rand_var_name'],
                                 print_characteristics=False,
                                 prob_rand_var_less_than=interactive_params['prob_rand_var_less_than'])
    return rand_vars


def gen_weibull_dist(_alpha, 
                     _lambda, 
                     size, 
                     round_to_nearest=None,
                     num_decimal_places=2,
                     min_val=None,
                     max_val=None,
                     interactive_params=None, 
                     logscale=False, 
                     transparent=True):
    '''Generates a Weibull distribution of random variable values.

    Weibull distributions often fir scenarios whose random variable values 
    (e.g. 'time until failure') are modelled by 'extreme value theory' (EVT) in
    that the values being predicted are more extreme than any previously recorded
    and, similar to the log-normal distribution, have a low mean but high variance
    and therefore a long tail/positive skew. Often use to predict time until
    failure.

    E.g. real-world scenarios: Paricle sizes generated by grinding, milling & 
    crushing operations, survival times after cancer diagnosis, light bulb 
    failure times, divorce rates, data centre arrival times, etc.

    Args:
        _alpha (int/float): Shape parameter. Describes slope of distribution.
        
            _alpha < 1: Probability of random variable occurring decreases as 
            values get higher. Occurs in systems with high 'infant mortality'
            in that e.g. defective items occur soon after t=0 and are therefore
            weeded out of the population early on.

            _alpha == 1: Special case of the Weibull distribution which reduces 
            the distribution to an exponential distribution.

            _alpha > 1: Probability of random variable value occurring increases
            with time (until peak passes). Occurs in systems with an 'aging'
            process whereby e.g. components are more likely to fail as time goes 
            on.
        _lambda (int/float): Weibull scale parameter. Use to shape distribution
            standard deviation.
        size (int): Number of random variable values to sample from distribution.
        interactive_params (dict): Dictionary of distribution parameter values
            (must provide if in interactive mode).
        logscale (bool): Whether or not plot should have logscale x-axis and bins.
        transparent (bool): Whether or not to make plot bins slightly transparent.
    
    Returns:
        list: random variable values generated by sampling from the distribution.

    '''
    rand_vars = (np.random.weibull(_alpha, size=size)) * _lambda

    # check min and max vals 
    min_rand_var, max_rand_var = min(rand_vars), max(rand_vars)
    counter = 0
    if min_val is None and max_val is None:
        # don't need to worry about any min or max allowed values
        pass
    elif min_val is not None and max_val is None:
        # ensure greater than min_val
        while min(rand_vars) < min_val:
            min_idx = np.argmin(np.array(rand_vars))
            rand_vars[min_idx] = (np.random.weibull(_alpha, size=1)) * _lambda
            counter += 1
            if counter > 10000:
                sys.exit('Dist to broad for required min-max range. Increase min-max range or reduce dist broadness.')
    elif min_val is None and max_val is not None:
        # ensure less than max val
        while max(rand_vars) > max_val:
            max_idx = np.argmax(np.array(rand_vars))
            rand_vars[max_idx] = (np.random.weibull(_alpha, size=1)) * _lambda
            counter += 1
            if counter > 10000:
                sys.exit('Dist to broad for required min-max range. Increase min-max range or reduce dist broadness.')
    elif min_val is not None and max_val is not None:
        # ensure less than max val and greater than min val
        while min(rand_vars) < min_val or max(rand_vars) > max_val:
            if min(rand_vars) < min_val:
                min_idx = np.argmin(np.array(rand_vars))
                rand_vars[min_idx] = (np.random.weibull(_alpha, size=1)) * _lambda
            if max(rand_vars) > max_val:
                max_idx = np.argmax(np.array(rand_vars))
                rand_vars[max_idx] = (np.random.weibull(_alpha, size=1)) * _lambda
            counter += 1
            if counter > 10000:
                sys.exit('Dist to broad for required min-max range. Increase min-max range or reduce dist broadness.')

    if round_to_nearest is not None:
        rand_vars = [x_round(val, round_to_nearest, num_decimal_places, min_val=min_val) for val in rand_vars]
    else:
        # no need to discretise
        pass

    if interactive_params is not None:
        # gen interactive plot
        data_description = stats.describe(rand_vars) 
        print('Characteristics of generated distribution:\n{}'.format(data_description))
        
        plot_dists.plot_val_dist(rand_vars, 
                                 xlim=interactive_params['xlim'], 
                                 logscale=logscale,
                                 num_bins=interactive_params['num_bins'],
                                 transparent=transparent,
                                 dist_fit_line='weibull', 
                                 rand_var_name=interactive_params['rand_var_name'],
                                 print_characteristics=False,
                                 prob_rand_var_less_than=interactive_params['prob_rand_var_less_than'])
    return rand_vars


def gen_named_val_dist(dist, 
                       params=None, 
                       interactive_plot=False,
                       size=150000, 
                       occurrence_multiplier=100,
                       return_data=False, 
                       round_to_nearest=None, 
                       num_decimal_places=2,
                       path_to_save=None,
                       plot_fig=False,
                       show_fig=False,
                       min_val=None,
                       max_val=None,
                       xlim=None,
                       logscale=False,
                       rand_var_name='Random Variable',
                       prob_rand_var_less_than=None,
                       num_bins=0,
                       print_data=False):
    '''Generates a 'named' (e.g. Weibull/exponential/log-normal/Pareto) distribution.

    Args:
        dist (str): One of the valid named distributions (e.g. 'weibull', 'lognormal',
            'pareto', 'exponential')
        params (dict): Corresponding parameter arguments of distribution (e.g. for
            Weibull, params={'_alpha': 1.4, '_lambda': 7000}). See individual
            name distribution function generators for more information.
        interactive_plot (bool): Whether or not you want to use the interactive
            functionality of this function in Jupyter notebook to visually
            shape your named distribution.
        size (int): Number of values to sample from generated distribution when
            generating random variable data.
        round_to_nearest (int/float): Value to round random variables to nearest.
            E.g. if round_to_nearest=0.2, will round each random variable to 
            nearest 0.2.
        num_decimal_places (int): Number of decimal places to random variable
            values. Need to explicitly state otherwise Python's floating point 
            arithmetic will cause spurious unique random variable value errors
            when discretising.
        occurrence_multiplier (int/float): When sampling random variables from
            distribution to create plot and random variable data, use this
            multiplier to determine number of data points to sample. A higher 
            value will cause the random variable data to match the probability
            distribution more closely, but will take longer to generate.
        path_to_save (str): Path to directory (with file name included) in which
            to save generated distribution. E.g. path_to_save='data/dists/my_dist'.
        plot_fig (bool): Whether or not to plot fig. If True, will return fig.
        show_fig (bool): Whether or not to plot and show fig. If True, will
            return and display fig.
        return_data (bool) Whether or not to return random variable data sampled
            from generated distribution.
        min_val (int/float): Minimum random variable value.
        max_val (int/float): Maximum random variable value.
        xlim (list): X-axis limits of plot. E.g. xlim=[0,10] to plot random
            variable values between 0 and 10.
        logscale (bool): Whether or not plot should have logscale x-axis and bins.
        rand_var_name (str): Name of random variable to label plot's x-axis.
        prob_rand_var_less_than (list): List of values for which to print the
            probability that a variable sampled randomly from the generated 
            distribution will be less than. This is useful for replicating 
            distributions from the literature. E.g. prob_rand_var_less_than=[3.7,5.8]
            will return the probability that a randomly chosen variable is less
            than 3.7 and 5.8 respectively.
        num_bins (int): Number of bins to use in plot. Default is 0, in which
            case the number of bins chosen will be automatically selected.
        print_data (bool): Whether or not to print extra information about the
            generated data.

    Returns:
        tuple: Tuple containing:
            - **prob_dist** (*dict*): Probability distribution whose key-value pairs are 
              random variable value-probability pairs. 
            - **rand_vars** (*list, optional*): Random variable values sampled from the 
              generated probability distribution. To return, set return_data=True.
            - **fig** (*matplotlib.figure.Figure, optional*): Probability density 
              and cumulative distribution function plot. To return, set show_fig=True 
              and/or plot_fig=True.

    '''
    if params is None:
        assert interactive_plot == True, 'if not using interactive, provide params dict'
        show_fig = False # dont show standard fig, only interactive
        # num_bins_widget = widgets.IntText(description='bins:',value=0,step=1,disabled=False)
    if params is not None:
        assert interactive_plot == False, 'cannot provide params dict if using interactive'

    if dist == 'exponential':
        if interactive_plot:
            _beta_exp_widget = widgets.FloatText(description='_beta:',value=0.1,step=0.01,disabled=False)
            rand_vars = interactive(gen_exponential_dist,
                                    {'manual': True},
                                    _beta=_beta_exp_widget,
                                    size=fixed(size),
                                    round_to_nearest=fixed(round_to_nearest),
                                    num_decimal_places=fixed(num_decimal_places),
                                    min_val=fixed(min_val),
                                    max_val=fixed(max_val),
                                    interactive_params=fixed({'xlim': xlim, 
                                                              'rand_var_name': rand_var_name,
                                                              'prob_rand_var_less_than': prob_rand_var_less_than,
                                                              'num_bins': num_bins}))
        else:
            rand_vars = gen_exponential_dist(_beta=params['_beta'], 
                                             size=size,
                                             round_to_nearest=round_to_nearest,
                                             num_decimal_places=num_decimal_places,
                                             min_val=min_val,
                                             max_val=max_val)

    elif dist == 'lognormal':
        if interactive_plot:
            _mu_lognormal_widget = widgets.FloatText(description='_mu:',value=0.1,step=0.05,disabled=False)
            _sigma_lognormal_widget = widgets.FloatText(description='_sigma:',value=1.0,step=0.1,disabled=False)
            rand_vars = interactive(gen_lognormal_dist,
                                    {'manual': True},
                                    _mu=_mu_lognormal_widget,
                                    _sigma=_sigma_lognormal_widget,
                                    size=fixed(size),
                                    round_to_nearest=fixed(round_to_nearest),
                                    num_decimal_places=fixed(num_decimal_places),
                                    min_val=fixed(min_val),
                                    max_val=fixed(max_val),
                                    interactive_params=fixed({'xlim': xlim, 
                                                              'rand_var_name': rand_var_name,
                                                              'prob_rand_var_less_than': prob_rand_var_less_than,
                                                              'num_bins': num_bins}))
        else:
            rand_vars = gen_lognormal_dist(_mu=params['_mu'],
                                           _sigma=params['_sigma'],
                                           size=size,
                                           round_to_nearest=round_to_nearest,
                                           num_decimal_places=num_decimal_places,
                                           min_val=min_val,
                                           max_val=max_val)

    elif dist == 'weibull':
        if interactive_plot:
            _alpha_weibull_widget = widgets.FloatText(description='_alpha:',value=5,step=0.1,disabled=False)
            _lambda_weibull_widget = widgets.FloatText(description='_lambda:',value=0.5,step=0.1,disabled=False)
            rand_vars = interactive(gen_weibull_dist,
                                    {'manual': True},
                                    _alpha=_alpha_weibull_widget,
                                    _lambda=_lambda_weibull_widget,
                                    size=fixed(size),
                                    round_to_nearest=fixed(round_to_nearest),
                                    num_decimal_places=fixed(num_decimal_places),
                                    min_val=fixed(min_val),
                                    max_val=fixed(max_val),
                                    interactive_params=fixed({'xlim': xlim, 
                                                              'rand_var_name': rand_var_name,
                                                              'prob_rand_var_less_than': prob_rand_var_less_than,
                                                              'num_bins': num_bins}))
        else:
            rand_vars = gen_weibull_dist(_alpha=params['_alpha'],
                                         _lambda=params['_lambda'],
                                         size=size,
                                         round_to_nearest=round_to_nearest,
                                         num_decimal_places=num_decimal_places,
                                         min_val=min_val,
                                         max_val=max_val)

    elif dist == 'pareto':
        if interactive_plot:
            _alpha_pareto_widget = widgets.FloatText(description='_alpha:',value=1.5,step=0.005,disabled=False)
            _mode_pareto_widget = widgets.FloatText(description='_mode:',value=3.0,step=0.1,disabled=False)
            rand_vars = interactive(gen_pareto_dist,
                                    {'manual': True},
                                    _alpha=_alpha_pareto_widget,
                                    _mode=_mode_pareto_widget,
                                    size=fixed(size),
                                    round_to_nearest=fixed(round_to_nearest),
                                    num_decimal_places=fixed(num_decimal_places),
                                    min_val=fixed(min_val),
                                    max_val=fixed(max_val),
                                    interactive_params=fixed({'xlim': xlim, 
                                                              'rand_var_name': rand_var_name,
                                                              'prob_rand_var_less_than': prob_rand_var_less_than,
                                                              'num_bins': num_bins}))
        else:
            rand_vars = gen_pareto_dist(_alpha=params['_alpha'],
                                        _mode=params['_mode'],
                                        size=size,
                                        round_to_nearest=round_to_nearest,
                                        num_decimal_places=num_decimal_places,
                                        min_val=min_val,
                                        max_val=max_val)

    elif dist == 'normal':
        if interactive_plot:
            _loc_normal_widget = widgets.FloatText(description='_loc:',value=1.5,step=0.005,disabled=False)
            _scale_widget = widgets.FloatText(description='_scale:',value=3.0,step=0.1,disabled=False)
            rand_vars = interactive(gen_normal_dist,
                                    {'manual': True},
                                    loc=_loc_normal_widget,
                                    scale=_scale_widget,
                                    size=fixed(size),
                                    round_to_nearest=fixed(round_to_nearest),
                                    num_decimal_places=fixed(num_decimal_places),
                                    min_val=fixed(min_val),
                                    max_val=fixed(max_val),
                                    interactive_params=fixed({'xlim': xlim, 
                                                              'rand_var_name': rand_var_name,
                                                              'prob_rand_var_less_than': prob_rand_var_less_than,
                                                              'num_bins': num_bins}))
        else:
            rand_vars = gen_normal_dist(loc=params['_loc'],
                                        scale=params['_scale'],
                                        size=size,
                                        round_to_nearest=round_to_nearest,
                                        num_decimal_places=num_decimal_places,
                                        min_val=min_val,
                                        max_val=max_val)

    elif dist == 'skewnorm':
        if interactive_plot:
            _a_skewnorm_widget = widgets.FloatText(description='_a:',value=0.0,step=0.1,disabled=False)
            _loc_skewnorm_widget = widgets.FloatText(description='_loc:',value=50.0,step=0.1,disabled=False)
            _scale_skewnorm_widget = widgets.FloatText(description='_scale:',value=10.0,step=0.1,disabled=False)
            rand_vars = interactive(gen_skewnorm_data,
                                    {'manual': True},
                                    a=_a_skewnorm_widget,
                                    loc=_loc_skewnorm_widget,
                                    scale=_scale_skewnorm_widget,
                                    num_samples=fixed(size),
                                    round_to_nearest=fixed(round_to_nearest),
                                    num_decimal_places=fixed(num_decimal_places),
                                    min_val=fixed(min_val),
                                    max_val=fixed(max_val),
                                    interactive_params=fixed({'xlim': xlim, 
                                                              'rand_var_name': rand_var_name,
                                                              'prob_rand_var_less_than': prob_rand_var_less_than,
                                                              'num_bins': num_bins}))
        else:
            rand_vars = gen_skewnorm_data(a=params['_a'],
                                          loc=params['_loc'],
                                          scale=params['_scale'],
                                          num_samples=size,
                                          round_to_nearest=round_to_nearest,
                                          num_decimal_places=num_decimal_places,
                                          min_val=min_val,
                                          max_val=max_val)



    else:
        raise Exception('Must provide valid name distribution to use')
   
    if interactive_plot:
        display(rand_vars)
        return rand_vars

    else:
        unique_vals, pmf = gen_discrete_prob_dist(rand_vars,
                                                  round_to_nearest=round_to_nearest)
        prob_dist = {unique_var: prob for unique_var, prob in zip(unique_vals, pmf)}

        if print_data:
            print('Prob dist:\n{}'.format(prob_dist))
        if path_to_save is not None:
            tools.pickle_data(path_to_save, prob_dist)
        if plot_fig or show_fig:
            min_prob = min(prob for prob in list(prob_dist.values()) if prob > 0)
            num_occurrences = [int(val*(1/min_prob)*occurrence_multiplier) for val in list(prob_dist.values())]
            data = convert_key_occurrences_to_data(unique_vals,num_occurrences)
            fig = plot_dists.plot_val_dist(rand_vars=data, 
                                           xlim=xlim,
                                           logscale=logscale,
                                           num_bins=num_bins,
                                           rand_var_name=rand_var_name,
                                           prob_rand_var_less_than=prob_rand_var_less_than,
                                           print_characteristics=False,
                                           show_fig=show_fig)
            if return_data:
                return prob_dist, rand_vars, fig
            else:
                return prob_dist, fig
        else:
            if return_data:
                return prob_dist, rand_vars
            else:
                return prob_dist


















