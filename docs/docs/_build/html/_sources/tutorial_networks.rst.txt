Networks
========

.. nbplot::
    >>> import trafpy.generator as tpg

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

A key feature of TrafPy is that it can generate traffic for any network. If your
network does not fall into one of the above networks (which is likely that it will not),
you should use the ``trafpy.generator.gen_arbitrary_network`` function to generate your network.
This will generate an arbitrary network given your number of end points, but will
format the network in a way recognised by TrafPy.

.. nbplot::

    >>> network = tpg.gen_arbitrary_network(num_eps=10)
