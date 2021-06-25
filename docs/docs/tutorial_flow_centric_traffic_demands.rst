Flow-Centric Traffic Demands
============================

.. nbplot::
    >>> import trafpy.generator as tpg

A single demand in a network can be considered as either a **flow** or a computation
graph (a **job**) whose dependencies (edges) may form flows. Both flow-centric
and job-centric network traffic demand generation and management are supported
by TrafPy.

A flow is some information being sent from a source node to a destination node
in a network (e.g. a data centre network).

Common flow demand characteristics include:

- size;
- interarrival time; and
- source-destination node distribution.

Using the value and node distribution generation functions you've seen so far,
you can use TrafPy to generate realistic flow demands. Later in this tutorial,
you will see how to use TrafPy's Jupyter Notebook tool to visually shape your
distributions such that they match real data/literature distributions. For now,
assume that you already know the distribution parameters you want. Consider
that you want to create 1,000 realistic data centre flows in a simple 5-node
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
    >>> node_dist = tpg.gen_multimodal_node_dist(eps=endpoints,num_skewed_nodes=1,show_fig=True)

You can then use your distributions to generate flow-centric demand data formatted
neatly into a single dictionary

.. nbplot::

    >>> flow_centric_demand_data = tpg.create_demand_data(eps=endpoints,node_dist=node_dist,flow_size_dist=flow_size_dist,interarrival_time_dist=interarrival_time_dist)

Don't forget to save your data as a pickle::

    tpg.pickle_data(data=flow_centric_demand_data,path_to_save='data/flow_centric_demand_data.pickle',overwrite=True,zip_data=True)

or as a csv::

    tpg.save_data_as_csv(data=flow_centric_demand_data,path_to_save='data/flow_centric_demand_data.csv',overwrite=True)

N.B. You can also re-load previously pickled data::
    
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

Where ``'establish'`` keys' values are binary values indicating whether the demand is a connection
establishment request (1) or a take-down request (0) for a given event. Specifying 
take-down requests is optional in TrafPy. If take-downs have been speified, then
there will be ``2 * num_demands`` events in the demand data dictionary, otherwise
there will be ``num_demands`` events.
