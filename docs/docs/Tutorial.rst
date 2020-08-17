Tutorial
========
This guide can help you start with TrafPy.






TrafPy Generator
----------------
Import the TrafPy generator package.

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
    
    >>> prob_dist, rand_vars, fig = tpg.gen_uniform_val_dist(min_val=0, max_val=100, round_to_nearest=1, return_data=True, show_fig=True, num_bins=101)

TrafPy probability distributions are defined as Python dictionaries with
value-probability key-value pairs

.. nbplot::

    >>> print('Uniform probability distribution:\n{}'.format(prob_dist))
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

and the *Pareto distribution*

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
Network traffic travels from a **source** node to a **destination** node.
Source-destination nodes are **endpoints** in a network

.. nbplot::

    >>> endpoints = ['server_'+str(i) for i in range(5)]

How regularly each node is selected as a source or destination is determined by a
**node distribution probability matrix**. The most simple node distribution
is the **uniform distribution**

.. nbplot::
    
    >>> node_dist, fig = tpg.gen_uniform_node_dist(eps=endpoints,show_fig=True)

Since different endpoint nodes in a network likely have different hardware
capabilities, network node distributions are rarely uniform. Instead, some nodes
become 'hot nodes' and are requested more than others, forming a **multimodal
node distribution**

.. nbplot::

    >>> node_dist, fig = tpg.gen_multimodal_node_dist(eps=endpoints,skewed_nodes=['server_2'],show_fig=True)

Instead of certain *nodes* being requested more regularly, sometimes certain
*node pairs* in the network might be skewed, forming a **multimodal node pair**
distribution

.. nbplot::

    >>> node_dist, fig = tpg.gen_multimodal_node_pair_dist(eps=endpoints,skewed_pairs=[['server_1','server_3'],['server_4','server_2']], show_fig=True)

Different networks have different node distributions. Sometimes you may want a 
simple uniform distribution, or a slightly skewed distribution, or certain nodes 
being heavily in demand, or certain node pairs being heavily in demand. Furthermore,
you may want all of the above, but may also want to specify certain things yourself
(e.g. which specific nodes/pairs to bias, how high demand they're in, how many nodes
are in high demand etc.), or you may want these specifics to be randomly generated.
The above functions handle all of the above functionality. See their documentation
for further details.

You can create any size of node distribution you like to fit any network

.. nbplot::

    >>> endpoints = ['server_'+str(i) for i in range(64)]
    >>> node_dist, fig = tpg.gen_multimodal_node_pair_dist(eps=endpoints,show_fig=True)

You can then sample from your node distribution to generate source-destination 
pairs

.. nbplot::

    >>> sn, dn = tpg.gen_node_demands(eps=endpoints,node_dist=node_dist,num_demands=1000)





Networks
^^^^^^^^

By definition, a network is a collection of nodes (vertices) which together
form pairs of nodes connected by links (edges). Some or all of these nodes can 
act as **sources** and **destinations** for network traffic **demands**. Such 
network nodes are referred to as **endpoints**. Endpoints might be separated by 
multiple links and nodes, some of which may be endpoints and some not. 

Generate a simple 5-node network

.. nbplot::

    >>> network = tpg.gen_simple_network(ep_label='server', show_fig=True)

or the 14-node NSFNET network

.. nbplot::

    >>> network = tpg.gen_nsfnet_network(ep_label='server', show_fig=True)

or a fat-tree network

.. nbplot::

    >>> network = tpg.gen_fat_tree(k=4, show_fig=True)

A single demand in a network can be considered as either a **flow** or a computation
graph (a **job**) whose dependencies (edges) may form flows. Both flow-centric
and job-centric network traffic demand generation and management are supported
by TrafPy.



Flow-Centric Traffic Demands
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A flow is some information being sent from a source node to a destination node
in a network (e.g. a data centre network).

Common flow demand characteristics include:

- size;
- interarrival time; and
- source-destination node distribution.

Using the value and node distribution generation function you've seen so far,
you can use TrafPy to generate realistics flow demands. Later in this tutorial,
you will see how to use TrafPy's Jupyter Notebook tool to visually shape your
distributions such that they match real data/literature distributions. For now,
assume that we already know the distribution parameters we want. Consider
that we want to create 1,000 realistic data centre flows in a simple 5-node
network

.. nbplot::

    >>> num_demands = 1000
    >>> network = tpg.gen_simple_network(ep_label='endpoint', show_fig=True)

You could start by defining the flow size distribution

.. nbplot::

    >>> flow_size_dist, _ = tpg.gen_named_val_dist(dist='weibull',params={'_alpha': 1.4, '_lambda': 7000},show_fig=True,rand_var_name='Flow Size',logscale=True,round_to_nearest=1,xlim=[1e2,1e12])

then the flow interarrival time distribution

.. nbplot::

    >>> interarrival_time_dist, _ = tpg.gen_named_val_dist(dist='lognormal',params={'_mu': 7.4, '_sigma': 2},show_fig=True,rand_var_name='Interarrival Time',logscale=True,round_to_nearest=1,xlim=[1e1,1e6])

and then the source-destination node distribution

.. nbplot::
    
    >>> endpoints = network.graph['endpoints']
    >>> node_dist, _ = tpg.gen_multimodal_node_dist(eps=endpoints,num_skewed_nodes=1,show_fig=True)

You can then use your distributions to generate flow-centric demand data formatted
neatly into a single dictionary

.. nbplot::

    >>> flow_centric_demand_data = tpg.create_demand_data(num_demands=num_demands,eps=endpoints,node_dist=node_dist,flow_size_dist=flow_size_dist,interarrival_time_dist=interarrival_time_dist)

Don't forget to save your data as a pickle

.. nbplot:: 
    
    >>> tpg.pickle_data(data=flow_centric_demand_data,path_to_save='data/flow_centric_demand_data.pickle',overwrite=True,zip_data=True)

or as a csv

.. nbplot::

    >>> tpg.save_data_as_csv(data=flow_centric_demand_data,path_to_save='data/flow_centric_demand_data.csv',overwrite=True)

N.B. You can also re-load previously pickled data

.. nbplot::
    
    >>> flow_centric_demand_data = tpg.unpickle_data(path_to_load='data/flow_centric_demand_data.pickle',zip_data=True)

TrafPy flow-centric demand data dictionaries are organised as::

    {
        'flow_id': ['flow_0', ..., 'flow_n'],
        'sn': [flow_0_sn, ..., flow_n_sn],
        'dn': [flow_0_dn, ..., flow_n_dn],
        'flow_size': [flow_0_size, ..., flow_n_size],
        'event_time': [event_time_flow_0, ..., event_time_flow_n],
        'establish': [event_establish_flow_0, ..., event_establish_flow_1],
        'index': [index_flow_0, ..., index_flow_1]
    }

Where ``'establish'`` key's values are is binary values indicating whether the demand is a connection
establishment request (1) or a take-down request (0) for a given event. Specifying 
take-down requests is optional in TrafPy. If take-downs have been speified, then
there will be ``2 * num_demands`` events in the demand data dictionary, otherwise
there will be ``num_demands`` events.



Job-Centric Traffic Demands
^^^^^^^^^^^^^^^^^^^^^^^^^^^
A job is a task sent to a network (such as a data centre) to execute. Jobs are 
computation graphs made up of **operations** (ops). Jobs might be e.g. a Google
search query, generating a user's Facebook feed, performing a TensorFlow machine
learning task (e.g. backpropagation), etc.

In this context, an op is a data process ran on some machine where the result
is specified by a pre-determined rule/programme. Each op requires >= 0 tensors/
data objects as input, and produces >= tensors as output.

In a job computation graph, if an op v requires >= 1 input(s) produced by op u,
the ops will be connected by a directed edge, [u, v], representing the **dependency**
between the two ops. The edge attributes here are features of the tensor (e.g. 
size, source machine, destination machine, etc.).

In a data centre, when a job arrives, each op in the job is placed onto some
machine to execute the op. These ops might be placed all on one machine or, as
is often the case for many applications, spread out across machines according
to e.g. some heuristic. The **network** is used to pass the tensors around
between the machines executing the ops. These tensors/data objects flowing
between ops are **flows**. The flows of a given job might flow through the network
at the same time or at different times depending on e.g. scheduling decisions, constraints,
dependencies, etc.

.. note:: In a job graph, edges between ops represent 1 of 2 types of op dependency:

          - **Data dependency**: Op j can only begin when op i's output tensor(s)
            have arrived. Therefore, data dependencies become network flows *if*
            op j and op i are ran on separate network endpoints.
          - **Control dependency**: Op j can only begin when op i has finished.
            No data is exchanged, therefore control dependencies never become 
            network flows.

Common job demand characteristics include:

- job interarrival time;
- which machine each op in the job is placed on;
- number of ops in the job;
- run times of the ops;
- size of data dependencies (flows) between ops;
- ratio of control to data dependencies in job computation graph; and
- connectivity of job graph.

You can use the same value and node distributions as before to generate realistic
job demands. The only difference is that now we will pass additional arguments
into :func:`tpg.create_demand_data`. TrafPy will respond by generating job computation graphs
rather than flows as the demands in the returned dictionary.

Consider that we want to create 10 realistic data centre jobs in the same simple 
5-node network as before (but now omitting ``show_fig`` to save page space).

.. nbplot:: 

    >>> num_demands = 10
    >>> tpg.gen_simple_network(ep_label='endpoint')

You could start by definiing the flow size distribution of the flows inside the 
job graphs

.. nbplot::

    >>> flow_size_dist = tpg.gen_multimodal_val_dist(min_val=1,max_val=100,locations=[50],skews=[0],scales=[10],num_skew_samples=[10000],bg_factor=0,round_to_nearest=1,num_bins=34)

then the job interarrival time distribution

.. nbplot::

    >>> interarrival_time_dist = tpg.gen_multimodal_val_dist(min_val=1,max_val=1e8,locations=[1,1,3000,1,1800000,10000000],skews=[0,100,-10,10,50,6],scales=[0.1,62,2000,7500,3500000,20000000],num_skew_samples=[800,1000,2000,4000,4000,3000],bg_factor=0.025,round_to_nearest=1)

then the number of ops in each job

.. nbplot:: 

    >>> num_ops_dist = tpg.gen_multimodal_val_dist(min_val=50,max_val=200,locations=[100],skews=[0.05],scales=[50],num_skew_samples=[10000],bg_factor=0.05,round_to_nearest=1)

and then the source-destination node (i.e. op machine placement) distribution

.. nbplot::
    
    >>> endpoints = network.graph['endpoints']
    >>> node_dist, _ = tpg.gen_multimodal_node_dist(eps=endpoints,num_skewed_nodes=1)

You can then use your distributions to generate your job-centric demand data
returned neatly into a single dictionary

.. nbplot::

    >>> job_centric_demand_data = tpg.create_demand_data(num_demands=num_demands,eps=endpoints,node_dist=node_dist,flow_size_dist=flow_size_dist,interarrival_time_dist=interarrival_time_dist,num_ops_dist=num_ops_dist,c=1.5,use_multiprocessing=False)

Don't forget to save your data

.. nbplot:: 
    
    >>> tpg.pickle_data(data=job_centric_demand_data,path_to_save='data/job_centric_demand_data.pickle',overwrite=True,zip_data=True)

TrafPy job-centric demand data dictionaries are organised as::

    {
        'job_id': ['job_0', ..., 'job_n'],
        'job': [networkx_graph_job_0, ..., networkx_graph_job_n],
        'event_time': [event_time_job_0, ..., event_time_job_n],
        'establish': [event_establish_job_0, ..., event_establish_job_1],
        'index': [index_job_0, ..., index_job_1]
    }

Where the ``'job'`` key contains the list of job computation graphs with all
the embedded demand data. You can visualise the job computation graph(s):

.. nbplot::

    >>> jobs = list(job_centric_demand_data['job'][0:2])
    >>> fig = tpg.draw_job_graphs(job_graphs=jobs,show_fig=True) 


Visually Shaping TrafPy Distributions
-------------------------------------
Up until now we have assumed we already knew all the parameters of each distribution
we have generated with TrafPy. But what if we want to replicate a distribution
which has either not been produced in TrafPy before or has not provided open-access
data? TrafPy has a useful interactive Jupyter-Notebook which integrates with
all of the above functions, allowing distributions to be visually shaped.










TrafPy Manager
--------------
