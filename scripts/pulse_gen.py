from trafpy.generator import Demand, DemandPlotter

import trafpy.generator as tpg

import time

import copy

import pathlib

from scipy.io import savemat

import sys

import os



if __name__ == '__main__':


    ################## SEED ##############################

    import numpy as np

    import random

    from trafpy.utils import seed_stochastic_modules_globally

    seed = 3

    seed_stochastic_modules_globally(default_seed=seed,

                                     numpy_module=np,

                                     random_module=random)

    #########################################################



    current_path = pathlib.Path().resolve().parents[1]

    path = str(current_path)+"/Traffic/Workloads/"



    print(path)



    sid = 1

    X = [1, 2, 4, 8]

    N = [i*64 for i in X] 

    print(N)

    for ns in N:

        endpoints = [str(i) for i in range(ns)]



        min_num_demands = ns*ns

        min_last_demand_arrival_time = 250

        jensen_shannon_distance_threshold = 0.1

        loads = 0.5

        NUM_DEMANDS_FACTOR = 5



        sk_nd = [0, 0.25,0.5,0.75, 0.5625,0.8125,0.3125, 0.5625,0.8125]

        sk_pr = [1, 0.64,0.64,0.64,0.16,0.16,0.32,0.32,0.32]



        SN = sk_nd[sid-1]

        SK = sk_pr[sid-1]



        # init network

        net = tpg.gen_arbitrary_network(ep_label=None, num_eps=ns, ep_capacity=100000)

        print(net.graph['max_nw_capacity'])



        # set any distributions you want to keep constant for all sets

        flow_size_dist = tpg.gen_named_val_dist(dist='weibull',

                                                params={'_alpha': 4.8, '_lambda': 4100},

                                                return_data=False,

                                                show_fig=False,

                                                round_to_nearest=1000)



        interarrival_time_dist = {0.125:1}



        # print('Generating uniform traffic...')

        # node_dist = tpg.gen_uniform_node_dist(eps=net.graph['endpoints'],

                                                #  show_fig=False)



        print('Generating skewed traffic...')

        node_dist = tpg.gen_multimodal_node_dist(eps=net.graph['endpoints'],

                                                skewed_nodes=[],

                                                skewed_node_probs=[SK/(SN*ns) for _ in range(int(SN*ns))],

                                                show_fig=False,

                                                plot_chord = False,

                                                num_skewed_nodes=int(SN*ns))

                                                





        print('Generating load {}...'.format(loads))

        print('Generating for N{}...'.format(ns))



        start = time.time()



        network_load_config = {'network_rate_capacity': net.graph['max_nw_capacity'], 

                                'ep_link_capacity': net.graph['ep_link_capacity'],

                                'target_load_fraction': loads}

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



        # save generated demands

        savemat(path+"uniform/seed{}_load{}_N{}_matlab_test.mat".format(seed, loads, ns), flow_centric_demand_data)



        end = time.time()

        print('Generated load {} in {} seconds.'.format(loads, end-start))
