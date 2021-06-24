Load and Rack Configurations
============================

Up to now, you have used TrafPy to create some number of demands. However,
it is often more useful to create an arbitrary number of demands such that a certain
**network load** is generated for an arbitrary network. Here, the **network capacity**
is the total *rate* at which information can be reliably transmitted over the network,
and the network load is the fraction of this capacity being requested during the
session. 

To specify the load and have the number of demands automatically generated,
the same functions you've seen above can be used, but now specifying the 
``network_load_config`` argument rather than ``num_demands``. E.g. to generate
demands that will request a 0.75 load of a network with a network capacity
of 6,000 Gbps:


# TEMPORARY COMMENT OUT OF BELOW CODE - BRING DEMO IN LATER WHEN FINALISED THIS FUNCTIONALITY
#.. nbplot::

#    >>> network_load_config = {'network_rate_capacity': 6000, 'target_load_fraction': 0.75}
#    >>> flow_centric_demand_data = tpg.create_demand_data(network_load_config=network_load_config,eps=endpoints,node_dist=node_dist,flow_size_dist=flow_size_dist,interarrival_time_dist=interarrival_time_dist)


Network endpoints/servers are often grouped into physically local clusters or 'racks'. 
Different networks may have different levels of inter- (between) and intra- (within) rack communication.
One way to specify this would be to set individual node pair probabilities with the ``gen_multimodal_node_pair_dist`` function
you've already seen, however this would be inconvenient and laborious. Instead, when using the above node distribution functions,
you can specify the ``rack_prob_config`` argument, which allows you to set the proportion of
traffic which should be inter-rack. TrafPy will then use your shaped node distribution
to create an adjusted node distribution which accounts for your specified rack probabilites.
For example, if you specify ``rack_prob_config`` in ``gen_uniform_node_dist``, you will not generate a perfectly
uniform node distribution as you would if you left ``rack_prob_config`` as ``None``,
but instead a node distribution with set inter- and intra-rack probabilities sampled
from a uniform distribution. You will need to specify which endpoints are
in which rack with a dictionary (this is automatically done for you if you
use one of the TrafPy networks). E.g. Making 20% of traffic inter-rack in a
fat-tree topology:

.. nbplot::

    >>> net = tpg.gen_fat_tree(k=3, N=2, num_channels=1)
    >>> fig = tpg.plot_network(net, draw_node_labels=True, network_node_size=1000)
    >>> print('Racks dict:\n{}'.format(net.graph['rack_to_ep_dict']))
    Racks dict:
    {'rack_0': ['server_0', 'server_1'], 'rack_1': ['server_2', 'server_3'], 
    'rack_2': ['server_4', 'server_5'], 'rack_3': ['server_6', 'server_7'], 
    'rack_4': ['server_8', 'server_9'], 'rack_5': ['server_10', 'server_11']}

    >>> rack_prob_config = {'racks_dict': net.graph['rack_to_ep_dict'], 'prob_inter_rack': 0.20}
    >>> node_dist, _ = tpg.gen_uniform_node_dist(net.graph['endpoints'], rack_prob_config=rack_prob_config, show_fig=True, print_data=False)

Making 90% of traffic inter-rack:

.. nbplot::

    >>> rack_prob_config = {'racks_dict': net.graph['rack_to_ep_dict'], 'prob_inter_rack': 0.90}
    >>> node_dist, _ = tpg.gen_uniform_node_dist(net.graph['endpoints'], rack_prob_config=rack_prob_config, show_fig=True, print_data=False)
