from trafpy.src.dists import plot_dists 

import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.inf)
from scipy.stats import skewnorm
from scipy import stats


def save_data(path, data):
    '''
    Save data to desired destination. Will always save as csv.

    Args:
    - path (str): path + filename (optional to ass .csv extension)
    - data: data to be saved as csv
    '''
    if path[-4:] != '.csv':
        path = path+'.csv'

    if type(data) == dict:
        try:
            df = pd.DataFrame(data)
        except ValueError:
            # dict values are scalars
            df = pd.DataFrame(data, index=[0])
    
    if type(data) == dict:
        df.to_csv(path)
    else:
        try:
            np.savetxt(path, data, delimiter=',')
        except TypeError:
            np.savetxt(path, data, delimiter=',', fmt='%s')


def convert_key_occurrences_to_data(keys, num_occurrences):
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
    # count number of times each var occurs in data
    counter_dict = {}
    for var in data:
        try:
            counter_dict[var] += 1
        except KeyError:
            # not yet encountered var val
            counter_dict[var] = 1

    return counter_dict

    

def x_round(x, round_to_nearest=1, num_decimal_places=2):
    '''
    Takes a rand var value x and rounds to nearest specified value (by default,
    will round x to nearest integer)
    '''
    factor = round(1/round_to_nearest, num_decimal_places)
    rounded = round(round(x*factor)/factor, num_decimal_places)

    return rounded
    

def gen_uniform_val_dist(min_val,
                         max_val,
                         round_to_nearest=None,
                         num_decimal_places=2,
                         path_to_save=None,
                         plot_fig=False,
                         show_fig=False,
                         xlim=None,
                         logscale=False,
                         rand_var_name='Random Variable',
                         prob_rand_var_less_than=None,
                         num_bins=0,
                         print_data=False):
    if round_to_nearest is None:
        # assume separation between vals is 1 unit
        separation = 1
    else:
        # separation between vals is same as round to nearest
        separation = round_to_nearest

    unique_vals = [val for val in np.arange(min_val, max_val+separation, separation)]
    if round_to_nearest is not None:
        unique_vals = [x_round(val,round_to_nearest,num_decimal_places) for val in unique_vals]
    else:
        # no need to discretise
        pass
    probabilities = [1/len(unique_vals) for _ in range(len(unique_vals))]

    prob_dist = {unique_val: prob for unique_val, prob in zip(unique_vals, probabilities)}
    
    if print_data:
        print('Prob dist:\n{}'.format(prob_dist))
    if path_to_save is not None:
        val_dists.save_data(path, node_dist)
    if plot_fig or show_fig:
        min_prob = min(prob for prob in list(prob_dist.values()) if prob > 0)
        num_occurrences = [int(val*(1/min_prob)) for val in list(prob_dist.values())]
        data = convert_key_occurrences_to_data(unique_vals,num_occurrences)
        fig = plot_dists.plot_val_dist(rand_vars=data, 
                                       xlim=xlim,
                                       logscale=logscale,
                                       rand_var_name=rand_var_name,
                                       prob_rand_var_less_than=prob_rand_var_less_than,
                                       num_bins=num_bins,
                                       show_fig=show_fig)
        return prob_dist, fig

    else:
        return prob_dist
    


def gen_multimodal_val_dist(min_val,
                            max_val,
                            locations = [40,80],
                            skews = [5.0,-5.0],
                            scales = [10,10],
                            num_skew_samples = [10000,650],
                            bg_factor=0.5,
                            round_to_nearest=None,
                            num_decimal_places=2,
                            path_to_save=None,
                            plot_fig=False,
                            show_fig=False,
                            xlim=None,
                            logscale=False,
                            rand_var_name='Random Variable',
                            prob_rand_var_less_than=None,
                            num_bins=0,
                            print_data=False):
    '''
    Generates multimodal val distribution for num_vals possible vals
    i.e. diff vals have diff probability of being chosen

    Args:
    - num_vals (int): number of possible vals
    - skewed_vals (list): list of vals that want to skew
    - skewed_val_probs (list): probabilites of these skewed vals being
    chosen
    - num_skewed_samples (list): number of samples in each skew. If have
    one skew with more samples than another, will have higher probability
    of demand falling within this particular skew
    - num_skewed_vals (int): number of vals to randomly skew
    - round_to_nearest (int or float): Value to round rand vars to nearest
    when discretising rand var values. E.g. if round_to_nearest=0.2, will 
    round each rand var to nearest 0.2
    - num_decimal_places (int): Number of decimal places for discretised
    rand vars. Need to explicitly state this because otherwise Python's
    floating point arithmetic will cause spurious unique random var vals
    when discretising

    Returns:
    - val_dist (array): multinomial val distribution
    - (optional) fig: plotted figure
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
    num_skews = len(locations)

    poss_vals = np.arange(min_val,max_val+separation,separation)
    baseline_probs = np.ones((len(poss_vals)))/len(poss_vals)
    baseline_vals = list(np.random.choice(poss_vals,
                                          size=num_bg_samples,
                                          p=baseline_probs))

    skew_data = []
    skew_data.append(baseline_vals)
    for skew_iter in range(num_skews):
        data = gen_skewnorm_data(a=skews[skew_iter],
                                 loc=locations[skew_iter],
                                 scale=scales[skew_iter],
                                 min_val=min_val,
                                 max_val=max_val,
                                 num_samples=num_skew_samples[skew_iter])
        skew_data.append(list(data))

    skew_data = [y for x in skew_data for y in x] # flatten
    
    unique_vals, pmf = gen_discrete_prob_dist(rand_vars=skew_data, 
                                              round_to_nearest=round_to_nearest,
                                              num_decimal_places=num_decimal_places)
    
    prob_dist = {unique_val: prob for unique_val, prob in zip(unique_vals, pmf)}
    
    if print_data:
        print('Prob dist:\n{}'.format(prob_dist))
    if path_to_save is not None:
        val_dists.save_data(path, node_dist)
    if plot_fig or show_fig:
        min_prob = min(prob for prob in list(prob_dist.values()) if prob > 0)
        num_occurrences = [int(val*(1/min_prob)) for val in list(prob_dist.values())]
        data = convert_key_occurrences_to_data(unique_vals,num_occurrences)
        fig = plot_dists.plot_val_dist(rand_vars=data, 
                                       xlim=xlim,
                                       logscale=logscale,
                                       rand_var_name=rand_var_name,
                                       prob_rand_var_less_than=prob_rand_var_less_than,
                                       num_bins=num_bins,
                                       show_fig=show_fig)
        return prob_dist, fig

    else:
        return prob_dist


def gen_skewnorm_data(a, loc, scale, min_val, max_val, num_samples):
    data = skewnorm(a, loc, scale).rvs(num_samples)
    for data_iter in range(len(data)):
        counter = 0
        while data[data_iter] < min_val or data[data_iter] > max_val:
            data[data_iter] = skewnorm(a, loc, scale).rvs(size=1)
            counter += 1
            if counter > 10000:
                sys.exit('scale too high for required max-min range')
    return list(data.astype(int))


def gen_rand_vars_from_discretised_dist(unique_vars, 
                                        probabilities, 
                                        num_demands,
                                        path_to_save=None):
    sampled_vars = np.random.choice(a=unique_vars, 
                                    size=num_demands,
                                    p=probabilities)
    
    if path_to_save is not None:
        save_data(path_to_save, sampled_vars)
    
    return sampled_vars


def gen_val_dist_data(val_dist, 
                      min_val, 
                      max_val, 
                      num_vals_to_gen,
                      path_to_save=None):
    '''
    Generates values between min_val and max_val 
    following the probability distribution given by val_dist

    Args:
    - num_vals_to_gen (int): number of flows to generate
    - val_dist (array): probability of selecting each possible val
    
    Returns:
    - vals (array)
    '''
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
        save_data(path_to_save, vals)
    
    return vals


def gen_discrete_prob_dist(rand_vars, 
                           round_to_nearest=None, 
                           num_decimal_places=2,
                           path_to_save=None):
    '''
    Takes rand var values, rounds to nearest value (specified as arg, defaults
    by not rounding at all) to discretise the data, and generates a 
    probability distribution for the data

    Args:
    - rand_vars (list): List of random variable values
    - round_to_nearest(int or float): Value to round rand vars to nearest when
    discretising rand var values. E.g. is round_to_nearest=0.2, will round each
    rand var to nearest 0.2
    - num_decimal_places (int): Number of decimal places for discretised rand vars.
    Need to explitly state this because otherwise Python's floating point 
    arithmetic will cause spurious unique random var values

    Returns:
    - xk (list): List of (discretised) unique random variable values that occurred
    - pmf (list): List of corresponding probabilities that each
    unique value in xk occurs
    '''
    if round_to_nearest is not None:
        # discretise vars
        discretised_rand_vars = [x_round(rand_var,round_to_nearest,num_decimal_places) for rand_var in rand_vars]  
    else:
        # no further discretisation required
        discretised_rand_vars = rand_vars

    # count number of times each var occurs in data
    counter_dict = {}
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
        save_data(path_to_save, prob_dist)

    return xk, pmf


def gen_exponential_dist(_beta, size):
    rand_vars = np.random.exponential(_beta,size=size)
    return rand_vars


def gen_lognormal_dist(_mu, _sigma, size):
    rand_vars = stats.lognorm.rvs(s=_sigma, scale=math.exp(_mu), size=size)
    return rand_vars


def gen_pareto_dist(_alpha, _mode, size):
    rand_vars = stats.pareto.rvs(b=_alpha, loc=0, scale=_mode, size=size)
    return rand_vars


def gen_weibull_dist(_alpha, _lambda, size):
    rand_vars = (np.random.weibull(_alpha, size=size)) * _lambda
    return rand_vars


def gen_named_val_dist(dist, 
                       params, 
                       size=30000, 
                       return_data=True, 
                       round_to_nearest=None, 
                       path_to_save=None,
                       plot_fig=False,
                       show_fig=False,
                       xlim=None,
                       logscale=False,
                       rand_var_name='Random Variable',
                       prob_rand_var_less_than=None,
                       num_bins=0,
                       print_data=False): 
    if dist == 'exponential':
        rand_vars = gen_exponential_dist(_beta=params['_beta'], 
                                         size=size)
    elif dist == 'lognormal':
        rand_vars = gen_lognormal_dist(_mu=params['_mu'],
                                       _sigma=params['_sigma'],
                                       size=size)
    elif dist == 'weibull':
        rand_vars = gen_weibull_dist(_alpha=params['_alpha'],
                                     _lambda=params['_lambda'],
                                     size=size)
    elif dist == 'pareto':
        rand_vars = gen_pareto_dist(_alpha=params['_alpha'],
                                    _mode=params['_mode'],
                                    size=size)
    else:
        sys.exist('Must provide valid name distribution to use')
    

    unique_vals, pmf = gen_discrete_prob_dist(rand_vars,
                                              round_to_nearest=round_to_nearest)
    prob_dist = {unique_var: prob for unique_var, prob in zip(unique_vals, pmf)}

    if print_data:
        print('Prob dist:\n{}'.format(prob_dist))
    if path_to_save is not None:
        val_dists.save_data(path, node_dist)
    if plot_fig or show_fig:
        min_prob = min(prob for prob in list(prob_dist.values()) if prob > 0)
        num_occurrences = [int(val*(1/min_prob)) for val in list(prob_dist.values())]
        data = convert_key_occurrences_to_data(unique_vals,num_occurrences)
        fig = plot_dists.plot_val_dist(rand_vars=data, 
                                       xlim=xlim,
                                       logscale=logscale,
                                       rand_var_name=rand_var_name,
                                       prob_rand_var_less_than=prob_rand_var_less_than,
                                       num_bins=num_bins,
                                       show_fig=show_fig)
        if return_data:
            return rand_vars, fig
        else:
            return prob_dist, fig
    else:
        if return_data:
            return rand_vars
        else:
            return prob_dist


















