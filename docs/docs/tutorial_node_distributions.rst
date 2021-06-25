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
    >>> node_dist, fig = tpg.gen_multimodal_node_pair_dist(eps=endpoints,show_fig=True)

Once you have your node probability distribution, you can use it to generate 
as many source-destination node pairs as you like

.. nbplot::

    >>> sn, dn = tpg.gen_node_demands(eps=endpoints,node_dist=node_dist,num_demands=1000)


