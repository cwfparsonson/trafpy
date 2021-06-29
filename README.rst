TrafPy
======

TrafPy is a Python package for the generation, management and standardisation of network traffic.

**Documentation**: https://trafpy.readthedocs.io/ 

Example
-------
Generate simple flow-centric traffic for an arbitrary network you want to put under 10% traffic load
with 16 leaf node end points which have communication link capacities of 1,250 information units
per unit time. Save the traffic trace in a universally compatible file format
so you can import it into any simulation, emulation, or experimentation test bed
independently of ``TrafPy``.

.. code:: python

    import trafpy.generator as tpg
    import gzip, pickle

    # init network
    net = tpg.gen_arbitrary_network(num_eps=16, ep_capacity=1250)

    # define flow characteristic distributions
    node_dist = tpg.gen_uniform_node_dist(eps=net.graph['endpoints'])
    interarrival_time_dist = {1.0: 1.0}
    flow_size_dist = {1.0: 1.0}

    # define network load config
    network_load_config = {'network_rate_capacity': net.graph['max_nw_capacity'], 
                           'ep_link_capacity': net.graph['ep_link_capacity'],
                           'target_load_fraction': 0.1}

    # generate traffic demands
    demand_data = tpg.create_demand_data(eps=net.graph['endpoints'],
                                         node_dist=node_dist,
                                         flow_size_dist=flow_size_dist,
                                         interarrival_time_dist=interarrival_time_dist,
                                         network_load_config=network_load_config,
                                         jensen_shannon_distance_threshold=0.9)

    # save
    Path('data/').mkdir(parents=True)
    with gzip.open('data/demand_data.pickle', 'wb') as f:
        pickle.dump(demand_data, f)


See the `tutorial <https://trafpy.readthedocs.io/en/latest/tutorial.html>`_
and the `examples <https://github.com/cwfparsonson/trafpy/tree/master/examples>`_ directory
for more information.


Install
-------

Open Git Bash. Change the current working directory to the location where you want
to clone this `GitHub <https://github.com/cwfparsonson/trafpy>`_ project, and run::

    $ git clone https://github.com/cwfparsonson/trafpy

In the project's root directory, run::

    $ python setup.py install

Then, still in the root directory, install the required packages with either pip::

    $ pip install -r requirements/default.txt

or conda::

    $ conda install --file requirements/default.txt


You should then be able to import TrafPy into your Python script from any directory
on your machine

.. code:: python

    import trafpy


Contributing
------------
File an issue `here <https://github.com/cwfparsonson/trafpy/issues>`_  to report 
any bugs or new suggestions. Or, even better, for the repository on `GitHub <https://github.com/cwfparsonson/trafpy>`_ 
and create a pull request. If you want help making
a pull request or are new to git, ask on the contributing issue you raise and/or
see TrafPy's `contributing guide <https://trafpy.readthedocs.io/en/latest/Contribute.html>`_.


License
-------
TrafPy uses Apache License 2.0.






