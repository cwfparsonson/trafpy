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

    >>> network = tpg.gen_simple_network(ep_label='endpoint', show_fig=True)

You could start by defining the flow size distribution

.. nbplot::

    >>> flow_size_dist, _ = tpg.gen_named_val_dist(dist='weibull',params={'_alpha': 1.4, '_lambda': 7000},show_fig=True,rand_var_name='Flow Size',logscale=True,round_to_nearest=25,xlim=[1e2,1e12], num_bins=15)

then the flow interarrival time distribution

.. nbplot::

    >>> interarrival_time_dist, _ = tpg.gen_named_val_dist(dist='weibull', params={'_alpha': 0.9, '_lambda': 6000}, rand_var_name='Interarrival Time', min_val=1, round_to_nearest=25, show_fig=True, logscale=True, print_data=False, num_bins=15)

and then the source-destination node distribution

.. nbplot::
    
    >>> endpoints = network.graph['endpoints']
    >>> node_dist = tpg.gen_multimodal_node_dist(eps=endpoints,num_skewed_nodes=1,show_fig=True)

The network load refers to the overall amount of traffic received by the
network. This is commonly referred to as a load rate (information units
arriving per unit time) or as a load fraction (the fraction of the total
network capacity being requested for a given duration). ``TrafPy`` typically uses
the load fraction definition for load, therefore loads can be varied between 0
and 1.

A key feature of ``TrafPy`` is that you can generate any load for your custom
network. To do this, you should provide ``TrafPy`` with a ``network_load_config``
dictionary which tells ``TrafPy`` (1) the end point capacity of your network, (2)
the maximum capacity of your network, and (3) the overall load fraction you
would like ``TrafPy`` to generate for your network. Consider that you would like
``TrafPy`` to generate a 0.1 load traffic trace for your network (i.e. around 10%
of your total network capacity will be requested per unit time):

.. nbplot::
    >>> network_load_config = {'network_rate_capacity': network.graph['max_nw_capacity'], 'ep_link_capacity': network.graph['ep_link_capacity'], 'target_load_fraction': 0.1}

You can then use your distributions and load config to generate flow-centric demand data formatted
neatly into a single dictionary::


    flow_centric_demand_data = tpg.create_demand_data(eps=endpoints,node_dist=node_dist,flow_size_dist=flow_size_dist,max_num_demands=1000,interarrival_time_dist=interarrival_time_dist,network_load_config=network_load_config)

Don't forget to save your data as a pickle::

    tpg.pickle_data(data=flow_centric_demand_data,path_to_save='data/flow_centric_demand_data.pickle',overwrite=True,zip_data=True)

or as a csv::

    tpg.save_data_as_csv(data=flow_centric_demand_data,path_to_save='data/flow_centric_demand_data.csv',overwrite=True)

N.B. You can also re-load previously pickled data::
    
    flow_centric_demand_data = tpg.unpickle_data(path_to_load='data/flow_centric_demand_data.pickle',zip_data=True)

TrafPy flow-centric demand data dictionaries are organised as::

    {
        'flow_id': ['flow_0', ..., 'flow_n'],
        'sn': [flow_0_sn, ..., flow_n_sn],
        'dn': [flow_0_dn, ..., flow_n_dn],
        'flow_size': [flow_0_size, ..., flow_n_size],
        'event_time': [event_time_flow_0, ..., event_time_flow_n],
        'index': [index_flow_0, ..., index_flow_1]
    }

