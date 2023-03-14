=======================================================
 :globe_with_meridians: TrafPy :vertical_traffic_light:
=======================================================


--------------------------------------------------------------------------------------------------------------------------------------------

TrafPy is a Python package for the generation, management, and standardisation of network traffic.

**Paper**: `Traffic generation for benchmarking data centre networks <https://www.sciencedirect.com/science/article/pii/S1573427722000315>`_

**Documentation**: https://trafpy.readthedocs.io/ 

--------------------------------------------------------------------------------------------------------------------------------------------

Projects which have used TrafPy
===============================

If you have used TrafPy in your project, please send an email with a link to cwfparsonson@gmail.com
to have your work added to the below list!

* `Y. Liu et al., 'A Hybrid Beam Steering Free-Space and Fiber Based Optical Data Center Network', Journal of Lightwave Technology, 2023 <https://ieeexplore.ieee.org/abstract/document/10064081>`_

* `C. W. F. Parsonson et al., 'A Vectorised Packing Algorithm for Efficient Generation of Custom Traffic Matrices', OFC'23: Optical Fiber Communications Conference and Exhibition, 2023 <https://arxiv.org/abs/2302.09970>`_

* `C. W. F. Parsonson et al., 'Traffic Generation for Benchmarking Data Centre Networks', Optical Switching and Networking, 2022 <https://www.sciencedirect.com/science/article/pii/S1573427722000315>`_

* `J. L. Benjamin et al., 'Traffic Tolerance of Nanosecond Scheduling on Optical Circuit Switched Data Centre Network', OFC'22: Optical Fiber Communications Conference and Exhibition, 2022 <https://ieeexplore.ieee.org/abstract/document/9748332>`_

* `J. L. Benjamin et al., 'Benchmarking Packet-Granular OCS Network Scheduling for Data Center Traffic Traces', Photonic Networks and Devices, 2021 <https://opg.optica.org/abstract.cfm?uri=Networks-2021-NeW3B.3>`_


--------------------------------------------------------------------------------------------------------------------------------------------





Example
=======
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
=======

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
    
    
    
Citing this work
================
If you find this project or the associated paper useful, please cite our work::

    @article{parsonson2022traffic,
        title = {Traffic generation for benchmarking data centre networks},
        journal = {Optical Switching and Networking},
        volume = {46},
        pages = {100695},
        year = {2022},
        issn = {1573-4277},
        doi = {https://doi.org/10.1016/j.osn.2022.100695},
        url = {https://www.sciencedirect.com/science/article/pii/S1573427722000315},
        author = {Christopher W.F. Parsonson and Joshua L. Benjamin and Georgios Zervas},
    }


Contributing
============
File an issue `here <https://github.com/cwfparsonson/trafpy/issues>`_  to report 
any bugs or new suggestions. Or, even better, for the repository on `GitHub <https://github.com/cwfparsonson/trafpy>`_ 
and create a pull request. If you want help making
a pull request or are new to git, ask on the contributing issue you raise and/or
see TrafPy's `contributing guide <https://trafpy.readthedocs.io/en/latest/Contribute.html>`_.


License
=======
TrafPy uses Apache License 2.0.







