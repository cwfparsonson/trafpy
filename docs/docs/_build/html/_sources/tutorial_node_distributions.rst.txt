Node Distributions
==================
.. nbplot::
    >>> import trafpy.generator as tpg

Network traffic travels from a **source** node to a **destination** node.
Source-destination nodes are **endpoints** in a network

.. nbplot::

    >>> endpoints = ['server_'+str(i) for i in range(5)]

How regularly each node is selected as a source or destination is determined by a
**node distribution matrix**. The elements in this matrix might refer to probabilities,
but in TrafPy they usually refer to **load fractions** (i.e. what fraction of the overall
traffic arriving is requested by a particular node pair).

The most simple node distribution is the **uniform distribution**

.. nbplot::
    
    >>> node_dist, fig = tpg.gen_uniform_node_dist(eps=endpoints, show_fig=True) 

Since different endpoint nodes in a network likely have different hardware
capabilities, network node distributions are rarely uniform. Instead, some nodes
become 'hot nodes' and are requested more than others, forming a **multimodal
node distribution**

.. nbplot::

    >>> node_dist, fig = tpg.gen_multimodal_node_dist(eps=endpoints, skewed_nodes=['server_2'], show_fig=True)

Instead of certain *nodes* being requested more regularly, sometimes certain
*node pairs* in the network might be skewed, forming a **multimodal node pair**
distribution

.. nbplot::

    >>> node_dist, fig = tpg.gen_multimodal_node_pair_dist(eps=endpoints, skewed_pairs=[['server_1','server_3'], ['server_4','server_2']], show_fig=True)

N.B. The above graph plots are **chord diagrams**. Each network end point is a node,
and the colour of the node indicates how much of the overall traffic requestes
that particular end point. The width of edges between nodes indicates how much
traffic is travelling between a particular endpoint pair, with pair edges
below a certain load threshold being excluded from the plot for visual clarity. 
Chord diagrams are just an alternative way of visualising the more standard 2D
traffic matrix.

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
    >>> node_dist, fig = tpg.gen_multimodal_node_pair_dist(eps=endpoints, show_fig=True)

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
use one of the TrafPy networks). E.g. Making 10% of traffic inter-rack in a
fat-tree topology:

.. nbplot::

    >>> net = tpg.gen_fat_tree(k=4, n=8)
    >>> fig = tpg.plot_network(net, draw_node_labels=True, network_node_size=1000)
    >>> print('Racks dict:\n{}'.format(net.graph['rack_to_ep_dict']))
    Racks dict:
    {'rack_0': ['server_0', 'server_1', 'server_2', 'server_3'], 'rack_1':
    ['server_4', 'server_5', 'server_6', 'server_7'], 'rack_2': ['server_8',
    'server_9', 'server_10', 'server_11'], 'rack_3': ['server_12', 'server_13',
    'server_14', 'server_15']}

    >>> rack_prob_config = {'racks_dict': net.graph['rack_to_ep_dict'], 'prob_inter_rack': 0.10}
    >>> node_dist, _ = tpg.gen_uniform_node_dist(net.graph['endpoints'], rack_prob_config=rack_prob_config, show_fig=True, print_data=False)

Making 90% of traffic inter-rack:

.. nbplot::

    >>> rack_prob_config = {'racks_dict': net.graph['rack_to_ep_dict'], 'prob_inter_rack': 0.90}
    >>> node_dist, _ = tpg.gen_uniform_node_dist(net.graph['endpoints'], rack_prob_config=rack_prob_config, show_fig=True, print_data=False)

Once you have your node distribution, you can use it to generate 
as many source-destination node pairs as you like

.. nbplot::

    >>> sn, dn = tpg.gen_node_demands(eps=net.graph['endpoints'], node_dist=node_dist, num_demands=1000)


