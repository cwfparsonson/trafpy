import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF


def plot_node_dist(node_dist, 
                   node_to_index_dict, 
                   eps, 
                   add_labels=False, 
                   show_fig=False):
    fig = plt.figure()
    plt.matshow(node_dist, cmap='coolwarm')
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
                  show_fig=False):
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












