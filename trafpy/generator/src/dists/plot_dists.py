'''Module for plotting node and value distributions.'''

import numpy as np
import copy
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('tkagg')
from statsmodels.distributions.empirical_distribution import ECDF
from trafpy.generator.src import tools
from trafpy.generator.src.dists import val_dists 
from trafpy.generator.src.dists import node_dists 
from scipy import stats



def plot_node_dist(node_dist, 
                   eps=None,
                   node_to_index_dict=None,
                   add_labels=False, 
                   add_ticks=False,
                   show_fig=False):
    '''Plots network node demand probability distribution as a 2D matrix.

    Args:
        node_dist (list or 2d numpy array): Source-destination pair probabilities 
            of being chosen. Must be either a 2d numpy matrix of probabilities or 
            a 1d list/array of node pair probabilities.
        eps (list): List of node endpoint labels.
        node_to_index_dict (dict): Maps node labels (keys) to integer indices (values).
        add_labels (bool): Whether or not to node labels to plot.
        add_ticks (bool): Whether or not to add ticks to x- and y-axis.
        show_fig (bool): Whether or not to plot and show fig. If True, will
            return and display fig.

    Returns:
        matplotlib.figure.Figure: node distribution plotted as a 2d matrix. 

    '''
    if type(node_dist[0]) == str and eps is None:
        eps = list(set(node_dist))
    else:
        assert eps is not None, 'must provide list of end points as arg if node_dist contains no endpoint labels.'

    if node_to_index_dict is None:
        _,_,node_to_index_dict,_=tools.get_network_params(eps) 
    fig = plt.figure()
    plt.matshow(node_dist, cmap='YlOrBr')
    cbar = plt.colorbar()
    if add_labels == True:
        for (i, j), z in np.ndenumerate(node_dist):
            plt.text(j, 
                     i, 
                     '{}'.format(z), 
                     ha='center', 
                     va='center',
                     bbox = dict(boxstyle='round', 
                     facecolor='white', 
                     edgecolor='0.3'))
    plt.xlabel('Destination (Node #)')
    plt.ylabel('Source (Node #)')
    cbar.ax.set_ylabel('Probability', rotation=270, x=0.5)
    if add_ticks:
        plt.xticks([node_to_index_dict[node] for node in eps])
        plt.yticks([node_to_index_dict[node] for node in eps])
    
    if show_fig:
        plt.show()

    return fig



def plot_val_dist(rand_vars, 
                  dist_fit_line=None, 
                  xlim=None, 
                  logscale=False,
                  transparent=False,
                  rand_var_name='Random Variable', 
                  prob_rand_var_less_than=None,
                  num_bins=0,
                  plot_cdf=True,
                  show_fig=False):
    '''Plots (1) probability distribution and (2) cumulative distribution function.
    
    Args:
        rand_vars (list): Random variable values.
        dist_fit_line (str): Line to fit to named distribution. E.g. 'exponential'.
            If not plotting a named distribution, leave as None.
        xlim (list): X-axis limits of plot. E.g. xlim=[0,10] to plot random
            variable values between 0 and 10.
        logscale (bool): Whether or not plot should have logscale x-axis and bins.
        transparent (bool): Whether or not to make plot bins slightly transparent.
        rand_var_name (str): Name of random variable to label plot's x-axis.
        num_bins (int): Number of bins to use in plot. Default is 0, in which
            case the number of bins chosen will be automatically selected.
        plot_cdf (bool): Whether or not to plot the CDF as well as the probability
            distribution.
        show_fig (bool): Whether or not to plot and show fig. If True, will
            return and display fig.
    
    Returns:
        matplotlib.figure.Figure: node distribution plotted as a 2d matrix. 

    '''
    if num_bins==0:
        histo, bins = np.histogram(rand_vars,density=True,bins='auto')
    else:
        histo, bins = np.histogram(rand_vars,density=True,bins=num_bins)
    if transparent:
        alpha=0.30
    else:
        alpha=1.0
    
    # PROBABILITY DENSITY
    fig = plt.figure(figsize=(15,5))
    plt.style.use('ggplot')
    plt.subplot(1,2,1)
    if logscale:
        ax = plt.gca()
        ax.set_xscale('log')
        logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
        plotbins = logbins
    else:
        plotbins = bins
    
    plt.hist(rand_vars,
             bins=plotbins,
             align='mid',
             color='tab:red',
             edgecolor='tab:red',
             alpha=alpha)
    
    if dist_fit_line is None:
        pass
    elif dist_fit_line == 'exponential':
        loc, scale = stats.expon.fit(rand_vars, floc=0)
        y = stats.expon.pdf(plotbins, loc, scale)
    elif dist_fit_line == 'lognormal':
        shape, loc, scale = stats.lognorm.fit(rand_vars, floc=0)
        y = stats.lognorm.pdf(plotbins, shape, loc, scale)
    elif dist_fit_line == 'weibull':
        shape, loc, scale = stats.weibull_min.fit(rand_vars, floc=0)
        y = stats.weibull_min.pdf(plotbins, shape, loc, scale)
    elif dist_fit_line == 'pareto':
        shape, loc, scale = stats.pareto.fit(rand_vars, floc=0)
        y = stats.pareto.pdf(plotbins, shape, loc, scale)

    plt.xlabel(rand_var_name)
    plt.ylabel('Probability Density')
    try:
        plt.xlim(xlim)
    except NameError:
        pass
    
    if plot_cdf:
        # CDF
        # empirical hist
        plt.subplot(1,2,2)
        if logscale:
            ax = plt.gca()
            ax.set_xscale('log')
        else:
            pass
        n,bins_temp,patches = plt.hist(rand_vars, 
                                       bins=plotbins, 
                                       cumulative=True,
                                       density=True,
                                       histtype='step',
                                       color='tab:red',
                                       edgecolor='tab:red')
        patches[0].set_xy(patches[0].get_xy()[:-1])
        # theoretical line
        ecdf = ECDF(rand_vars)
        plt.plot(ecdf.x, ecdf.y, alpha=0.5, color='tab:blue')
        plt.xlabel(rand_var_name)
        plt.ylabel('CDF')
        try:
            plt.xlim(xlim)
        except NameError:
            pass
        plt.ylim(top=1)
    
    # PRINT ANY EXTRA ANALYSIS OF DISTRIBUTION
    if prob_rand_var_less_than is None:
        pass
    else:
        for prob in prob_rand_var_less_than:
            print('P(x<{}): {}'.format(prob, ecdf(prob)))
    
    if show_fig:
        plt.show()

    return fig


def plot_val_bar(x_values,
                 y_values,
                 ylabel='Random Variable',
                 ylim=None,
                 xlabel=None,
                 plot_x_ticks=True,
                 bar_width=0.35,
                 show_fig=False):
    '''Plots standard bar chart.'''
    x_pos = [x for x in range(len(x_values))]

    fig = plt.figure()
    plt.style.use('ggplot')

    plt.bar(x_pos, y_values, bar_width)

    plt.ylabel(ylabel)
    if plot_x_ticks:
        plt.xticks(x_pos, (x_val for x_val in x_values))
    if xlabel is not None:
        plt.xlabel(xlabel)

    try:
        plt.ylim(ylim)
    except NameError:
        pass

    if show_fig:
        plt.show()

    return fig






def plot_val_stacked_bar(plot_dict={},
                         ylabel='Random Variable',
                         ylim=None,
                         bar_width=0.35,
                         show_fig=False):
    '''Plots stacked bar chart.

    E.g. plot_dict given should be of the form:

    plot_dict= {'class_1': {'x_values': ['Uni DCN', 'Private DCN', 'Cloud DCN'], 'y_values': [20, 40, 80]},
                'class_2': {'x_values': ['Uni DCN', 'Private DCN', 'Cloud DCN'], 'y_values': [80, 60, 20]}}
    ylim=[0,100]

    '''


    keys = list(plot_dict.keys())
    num_vals = len(plot_dict[keys[0]]['x_values'])
    for key in keys:
        if len(plot_dict[key]['x_values']) != num_vals or len(plot_dict[key]['y_values']) != num_vals:
            raise Exception('Must have equal number of x and y values to plot if want to stack bars.')
    x_pos = [x for x in range(num_vals)]


    fig = plt.figure()
    plt.style.use('ggplot')

    plots = {}
    # plot first class
    curr_bottom = None # init bottom y coords of bar to plot
    for _class in plot_dict.keys():
        plots[_class] = plt.bar(x_pos, plot_dict[_class]['y_values'], bar_width, bottom=curr_bottom)
        # update bottom y coords for next bar
        curr_bottom = plot_dict[_class]['y_values']

    plt.ylabel(ylabel)
    plt.xticks(x_pos, (x_val for x_val in plot_dict[_class]['x_values']))
    plt.legend((plots[key][0] for key in list(plots.keys())), (_class for _class in (plot_dict.keys())))

    try:
        plt.ylim(ylim)
    except NameError:
        pass

    if show_fig:
        plt.show()

    return fig














