Value Distributions
===================

To import the TrafPy generator, run:

.. nbplot::
    >>> import trafpy.generator as tpg

The most simple probability distribution for random variable values is the 
**uniform distribution**, where each random variable value has an equal probability
of occurring:

.. nbplot::
    
    >>> prob_dist, rand_vars, fig = tpg.gen_uniform_val_dist(min_val=0, max_val=100, round_to_nearest=1, return_data=True, show_fig=True, num_bins=101)


Demand characteristics of real network traffic patterns are rarely uniform. However,
they can often be described by certain well-defined **named distributions**. These
named distributions are themselves characterised by just a few parameters, making them
easy to reproduce.

Named distributions supported by TrafPy include the *exponential distribution*

.. nbplot::

    >>> prob_dist, rand_vars, fig = tpg.gen_named_val_dist(dist='exponential', params={'_beta': 1.0}, return_data=True, show_fig=True, xlim=[0,10], num_bins=101)

.. image:: images/exponential_dist_param_composite.png
    :align: center

the *log-normal distribution*

.. nbplot::

    >>> prob_dist, rand_vars, fig = tpg.gen_named_val_dist(dist='lognormal', params={'_mu': 0, '_sigma': 1.0}, return_data=True, show_fig=True, xlim=[0,5], num_bins=10000)

.. image:: images/lognormal_dist_param_composite.png
    :align: center

the *Weibull distribution*

.. nbplot::

    >>> prob_dist, rand_vars, fig = tpg.gen_named_val_dist(dist='weibull', params={'_alpha': 1.5, '_lambda': 1.0}, return_data=True, show_fig=True, xlim=[0,3.5], num_bins=101)

.. image:: images/weibull_dist_param_composite.png
    :align: center

and the *Pareto distribution*

.. nbplot::

    >>> prob_dist, rand_vars, fig = tpg.gen_named_val_dist(dist='pareto', params={'_alpha': 3.0, '_mode': 1.0}, return_data=True, show_fig=True, xlim=[0,5], num_bins=101)

.. image:: images/pareto_dist_param_composite.png
    :align: center

However, some demand characteristics cannot be accurately described by these named
distributions. Instead, they might be described by arbitrarily **skewed** distributions
which may or may not have some amount of **skewness** and/or **kurtosis**

.. nbplot::
    >>> prob_dist, rand_vars, fig = tpg.gen_skewnorm_val_dist(location=50, skew=-5, scale=10, return_data=True, show_fig=True, num_bins=15)

Note that setting ``skew=0`` simply generates a **normal** distribution with 
standard deviation ``scale`` and mean ``location``

.. nbplot::
    >>> prob_dist, rand_vars, fig = tpg.gen_skewnorm_val_dist(location=50, skew=0, scale=10, return_data=True, show_fig=True, num_bins=15)

You can also use TrafPy to generate **multimodal** distributions with an arbitrary
number of modes

.. nbplot::

    >>> prob_dist, rand_vars, fig = tpg.gen_multimodal_val_dist(min_val=10,max_val=7000,locations=[20,4000],skews=[6,-1],scales=[150,1500],num_skew_samples=[10000,650],bg_factor=0.05,return_data=True,show_fig=True,logscale=True,xlim=[10,10000],num_bins=18)

Later in this tutorial, you will see how to visually shape a multimodal distribution
using TrafPy, allowing for almost any distribution to be generated.

Once you have your value distribution, you can use it to generate as many
random variable values as you like

.. nbplot::

    >>> rand_vars = tpg.gen_rand_vars_from_discretised_dist(unique_vars=list(prob_dist.keys()),probabilities=list(prob_dist.values()),num_demands=1000)
