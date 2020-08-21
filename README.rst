TrafPy
======

TrafPy is a Python package for the generation, management and standardisation of network traffic.

**Documentation**: https://trafpy.readthedocs.io/ 

Example
-------
Generate realistic flow-centric traffic demands for a data centre network and 
use the traffic to test the performance of the *Shortest Remaining Processing Time*
(SRPT) scheduling algorithm:

.. code:: python

    import trafpy.generator as tpg
    from trafpy.manager import Demand, RWA, SRPT, DCN
    import config

    # create demand data
    flow_size_dist = tpg.gen_named_val_dist(dist='weibull',
                                            params={'_alpha': 1.4, '_lambda': 7000},
                                            round_to_nearest=1)
    interarrival_time_dist = tpg.gen_named_val_dist(dist='lognormal',
                                                    params={'_mu': 7.4, '_sigma': 2},
                                                    round_to_nearest=1)
    network = tpg.gen_simple_network(ep_label=config.ENDPOINT_LABEL,
                                     num_channels=config.NUM_CHANNELS)
    node_dist = tpg.gen_multimodal_node_dist(eps=network.graph['endpoints'],
                                             num_skewed_nodes=1)
    flow_centric_demand_data = tpg.create_demand_data(num_demands=config.NUM_DEMANDS,
                                                      eps=network.graph['endpoints'],
                                                      node_dist=node_dist,
                                                      flow_size_dist=flow_size_dist,
                                                      interarrival_time_dist=interarrival_time_dist)

    # init manager
    demand = Demand(demand_data=flow_centric_demand_data)
    rwa = RWA(tpg.gen_channel_names(config.NUM_CHANNELS), 
              config.NUM_K_PATHS)
    scheduler = SRPT(network, rwa, slot_size=config.SLOT_SIZE)
    env = DCN(network, 
              demand, 
              scheduler, 
              slot_size=config.SLOT_SIZE, 
              max_flows=config.MAX_FLOWS, 
              max_time=config.MAX_TIME)

    # run simulation
    for episode in range(config.NUM_EPISODES):
        print('\nEpisode {}/{}'.format(episode+1,config.NUM_EPISODES))
        observation = env.reset(config.LOAD_DEMANDS)
        while True:
            print('Time: {}'.format(env.curr_time))
            action = scheduler.get_action(observation)
            print('Action:\n{}'.format(action))
            observation, reward, done, info = env.step(action)
            if done:
                print('Episode finished.')
                break

    >>> env.get_scheduling_session_summary(print_summary=True)

::

    -=-=-=-=-=-=-= Scheduling Session Ended -=-=-=-=-=-=-=
    SUMMARY:
    ~* General Info *~
    Total session duration: 80000.0 time units
    Total number of generated demands (jobs or flows): 10
    Total info arrived: 61316.0 info units
    Load: 1.8257503573130063 info unit demands arrived per unit time (from first to last flow arriving)
    Total info transported: 61316.0 info units
    Throughput: 0.76645 info units transported per unit time

    ~* Flow Info *~
    Total number generated flows (src!=dst,dependency_type=='data_dep'): 10
    Time first flow arrived: 0.0 time units
    Time last flow arrived: 33584.0 time units
    Time first flow completed: 10000.0 time units
    Time last flow completed: 80000.0 time units
    Total number of demands that arrived and became flows: 10
    Total number of flows that were completed: 10
    Total number of dropped flows + flows in queues at end of session: 0
    Average FCT: 24527.6 time units
    99th percentile FCT: 73821.64 time units

See the `documentation's tutorial <https://trafpy.readthedocs.io/en/latest/Tutorial.html>`_
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

    import trafpy.generator as tpg
    from trafpy.manager import Demand, RWA, SRPT, DCN


Contributing
------------
File an issue `here <https://github.com/cwfparsonson/trafpy/issues>`_  to report 
any bugs or new suggestions. Or, even better, for the repository on `GitHub <https://github.com/cwfparsonson/trafpy>`_ 
and create a pull request. If you want help making
a pull request or are new to git, ask on the contributing issue you raise and/or
see TrafPy's `contributing guide <https://docs.dgl.ai/contribute.html>`_.


License
-------







