TrafPy
======

TrafPy is a Python package for the generation, management and standardisation of network traffic.

**Documentation**: https://trafpy.readthedocs.io/ 

Simple Example
--------------
Generate realistic flow-centric traffic demands for a data centre network and 
use the traffic to test the performance of the *Shortest Remaining Processing Time*
(SRPT) scheduling algorithm.

.. literalinclude:: readme_example.py
  :language: python
  :linenos:

.. code-block:: python
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
                                                      eps=endpoints,
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

    # analyse session
    env.get_scheduling_session_summary(print_summary=True)


