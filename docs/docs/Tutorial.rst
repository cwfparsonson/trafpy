Tutorial
========
This guide can help you start with TrafPy.






TrafPy Generator
----------------
Import the TrafPy generator module.

.. nbplot::

    >>> import trafpy.generator as tpg 

Network traffic patterns can be characterised by probability distributions. By
accurately describing a probability distribution, one can sample from it to generate
arbitrary amounts of realistic network traffic. 


Value Distributions
^^^^^^^^^^^^^^^^^^^
The most simple probability distribution for random variable values is the 
**uniform distribution**, where each random variable value has an equal probability
of occurring

.. nbplot::
    
    >>> prob_dist, rand_vars, fig = tpg.gen_uniform_val_dist(min_val=0, max_val=100, return_data=True, show_fig=True, num_bins=101)

TrafPy probability distributions are defined as Python dictionaries with
value-probability key-value pairs

.. nbplot::

    >>> print(Uniform probability distribution:\n{}'.format(prob_dist))
    Uniform probability distribution:
    {1: 0.01, 2: 0.01, 3: 0.01, 4: 0.01, 5: 0.01, 6: 0.01, 7: 0.01, 8: 0.01, 9: 0.01, 10: 0.01, 11: 0.01, 12: 0.01, 13: 0.01, 14: 0.01, 15: 0.01, 16: 0.01, 17: 0.01, 18: 0.01, 19: 0.01, 20: 0.01, 21: 0.01, 22: 0.01, 23: 0.01, 24: 0.01, 25: 0.01, 26: 0.01, 27: 0.01, 28: 0.01, 29: 0.01, 30: 0.01, 31: 0.01, 32: 0.01, 33: 0.01, 34: 0.01, 35: 0.01, 36: 0.01, 37: 0.01, 38: 0.01, 39: 0.01, 40: 0.01, 41: 0.01, 42: 0.01, 43: 0.01, 44: 0.01, 45: 0.01, 46: 0.01, 47: 0.01, 48: 0.01, 49: 0.01, 50: 0.01, 51: 0.01, 52: 0.01, 53: 0.01, 54: 0.01, 55: 0.01, 56: 0.01, 57: 0.01, 58: 0.01, 59: 0.01, 60: 0.01, 61: 0.01, 62: 0.01, 63: 0.01, 64: 0.01, 65: 0.01, 66: 0.01, 67: 0.01, 68: 0.01, 69: 0.01, 70: 0.01, 71: 0.01, 72: 0.01, 73: 0.01, 74: 0.01, 75: 0.01, 76: 0.01, 77: 0.01, 78: 0.01, 79: 0.01, 80: 0.01, 81: 0.01, 82: 0.01, 83: 0.01, 84: 0.01, 85: 0.01, 86: 0.01, 87: 0.01, 88: 0.01, 89: 0.01, 90: 0.01, 91: 0.01, 92: 0.01, 93: 0.01, 94: 0.01, 95: 0.01, 96: 0.01, 97: 0.01, 98: 0.01, 99: 0.01, 100: 0.01}

and the probability density plot is constructed by sampling random variables from the discrete probability distribution.

Demand characteristics of real network traffic patterns are rarely uniform. However,
they can often be described by certain well-defined **named distributions**. These
named distributions are themselves characterised by just a few parameters, making them
easy to reproduce.

Named distributions supported by TrafPy include the *exponential distribution*

.. nbplot::

    >>> prob_dist, rand_vars, fig = tpg.gen_named_val_dist(dist='exponential', params={'_beta': 1.0}, return_data=True, show_fig=True, xlim=[0,10])

the *log-normal distribution*

.. nbplot::

    >>> prob_dist, rand_vars, fig = tpg.gen_named_val_dist(dist='lognormal', params={'_mu': 0, '_sigma': 1.0}, return_data=True, show_fig=True, xlim=[0,5])

the *Weibull distribution*

.. nbplot::

    >>> prob_dist, rand_vars, fig = tpg.gen_named_val_dist(dist='weibull', params={'_alpha': 1.5, '_lambda': 1.0}, return_data=True, show_fig=True, xlim=[0,3.5])

and the *pareto distribution*

.. nbplot::

    >>> prob_dist, rand_vars, fig = tpg.gen_named_val_dist(dist='pareto', params={'_alpha': 3.0, '_mode': 1.0}, return_data=True, show_fig=True, xlim=[0,5])

However, some demand characteristics cannot be accurately described by these named
distributions. Instead, they are described by arbitrary **multimodal distributions**, 
which are distributions with more than one **mode** which may or may not have some amount
of **skewness** and/or **kurtosis**

.. nbplot::

    >>> prob_dist, rand_vars, fig = tpg.gen_multimodal_val_dist(min_val=10,max_val=7000,locations=[20,4000],skews=[6,-1],scales=[150,1500],num_skew_samples=[10000,650],bg_factor=0.05,return_data=True,show_fig=True,logscale=True,xlim=[10,10000],num_bins=18)

Later in this tutorial, we will see how to visually shape a multimodal distribution
using TrafPy, allowing for almost any distribution to be generated.


Node Distributions
^^^^^^^^^^^^^^^^^^





Networks
^^^^^^^^

By definition, a network is a collection of nodes (vertices) which together
form pairs of nodes connected by links (edges). Some or all of these nodes can 
act as **sources** and **destinations** for network traffic **demands**. Such 
network nodes are referred to as **endpoints**. Endpoints might be separated by 
multiple links and nodes, some of which may be endpoints and some not. 

Generate a simple 5-node network.

.. nbplot::

    >>> network = tpg.gen_simple_network(ep_label='server', show_fig=True)

A single demand in a network can be considered as either a **flow** or a computation
graph (a **job**) whose dependencies (edges) may form flows. Both flow-centric
and job-centric network traffic demand generation and management are supported
by TrafPy.



Flow-Centric Traffic Demands
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A flow is some information being sent from a source node to a destination node
in a network (e.g. a data centre network).

Common flow characteristics include:

- flow size







Job-Centric Traffic Demands
^^^^^^^^^^^^^^^^^^^^^^^^^^^







Visually Shaping TrafPy Distributions
-------------------------------------










TrafPy Manager
--------------
