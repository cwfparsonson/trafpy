from trafpy.generator import Demand, DemandPlotter
import trafpy.generator as tpg
from trafpy.utils import seed_stochastic_modules_globally

import time
import copy
from scipy.io import savemat

import numpy as np
import random

import cProfile
import pstats
import os



if __name__ == '__main__':
    # HOW TO USE TIME PROFILER:
    # 1. Generate a file called <name>.prof
    # 2. Transfer to /home/cwfparsonson/Downloads
    # 3. Run snakeviz /home/cwfparsonson/Downloads/<name>.prof to visualise

    seed = 0
    seed_stochastic_modules_globally(default_seed=seed,
                                     numpy_module=np,
                                     random_module=random)

    sid = 1
    # set vars
    # X = 8
    X = 2
    N = 64*X
    # X = 1
    # N = 32*X
    # path = r'Y:/Joshua/Traffic/'
    # path = 'Y:/Joshua/Traffic/'
    endpoints = [str(i) for i in range(N)]

    # min_num_demands = 100
    min_num_demands = N * N * 10
    # min_num_demands = int(300)
    # min_last_demand_arrival_time = 250
    min_last_demand_arrival_time = None
    # min_last_demand_arrival_time = N*N
    # loads = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    # loads = [0.1]
    # loads = [0.5]
    loads = [0.5]
    # jensen_shannon_distance_threshold = 0.1
    jensen_shannon_distance_threshold = 0.2
    # jensen_shannon_distance_threshold = 0.3
    # jensen_shannon_distance_threshold = 0.9
    NUM_DEMANDS_FACTOR = 5

    sk_nd = [0, 0.25,0.5,0.75, 0.5625,0.8125,0.3125, 0.5625,0.8125]
    sk_pr = [1, 0.64,0.64,0.64,0.16,0.16,0.32,0.32,0.32]

    SN = sk_nd[sid-1]
    SK = sk_pr[sid-1]
    # num_skewed_nodes=int(len(endpoints)*SN)

    # init network
    net = tpg.gen_arbitrary_network(ep_label=None, num_eps=N, ep_capacity=100000)
    print(f'Initialised network with {N} endpoints.')

    # # set any distributions you want to keep constant for all sets
    # flow_size_dist, _ = tpg.gen_named_val_dist(dist='weibull',
                                            # params={'_alpha': 4.8, '_lambda': 4100},
                                            # return_data=False,
                                            # show_fig=True,
                                            # round_to_nearest=1)
    # flow_size_dist = {50: 0.5, 100: 0.5}
    flow_size_dist = {1: 1}
    print(f'Initialised flow size distribution.')

    interarrival_time_dist = {0.125:1}
    print(f'Initialised interarrival time distribution.')

    # node_dist = tpg.gen_uniform_node_dist(eps=net.graph['endpoints'],
    #                                          show_fig=False)    

    # raise Exception()
    node_dist, _ = tpg.gen_multimodal_node_dist(eps=net.graph['endpoints'],
                                             skewed_nodes=[],
                                             skewed_node_probs=[SK/(SN*N) for _ in range(int(SN*N))],
                                             # show_fig=False,
                                             show_fig=True,
                                             num_skewed_nodes=int(SN*N))
    print(f'Initialised node distribution.')

    # init time profile
    profiler = cProfile.Profile()
    profiler.enable()

    for load in loads:
        print('Generating load {}...'.format(load))
        
        start = time.time()
        
        network_load_config = {'network_rate_capacity': net.graph['max_nw_capacity'], 
                               'ep_link_capacity': net.graph['ep_link_capacity'],
                               'target_load_fraction': load}
        flow_centric_demand_data = tpg.create_demand_data(eps=net.graph['endpoints'],
                                                          node_dist=node_dist,
                                                          flow_size_dist=flow_size_dist,
                                                          interarrival_time_dist=interarrival_time_dist,
                                                          network_load_config=network_load_config,
                                                          jensen_shannon_distance_threshold=jensen_shannon_distance_threshold,
                                                          min_num_demands=min_num_demands,
                                                          min_last_demand_arrival_time=min_last_demand_arrival_time,
                                                          check_dont_exceed_one_ep_load=True,
                                                          auto_node_dist_correction=True,
                                                          print_data=True)

        demand = Demand(flow_centric_demand_data, net.graph['endpoints'])
        # plotter = DemandPlotter(demand)
        # plotter.plot_node_dist(eps=net.graph['endpoints'])
        # plotter.plot_node_load_dists(eps=net.graph['endpoints'], ep_link_bandwidth=net.graph['ep_link_capacity'])
        
        # # save generated demands
        # savemat("custom/uniform/load{}_N{}_matlab_matrix.mat".format(load, N), flow_centric_demand_data)
        
        end = time.time()
        print('Generated load {} in {} seconds.'.format(load, end-start))

    # save time profile
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    save_dir = '/home/zciccwf/phd_project/projects/trafpy/scripts/speed_tests/time_profiles'
    file_name = 'time_profile'
    i = 0
    while os.path.exists(f'{save_dir}/{file_name}_{i}.prof'):
        i += 1
    stats.dump_stats(f'{save_dir}/{file_name}_{i}.prof')
    print(f'Saved time profile to {save_dir}/{file_name}_{i}.prof')
