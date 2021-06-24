Networks
========

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
