from trafpy.generator.src import tools
from trafpy.generator.src.dists import val_dists, node_dists, plot_dists

import numpy as np
import time
from collections import defaultdict # use for initialising arbitrary length nested dict
import json
import copy
import random
import math
import multiprocessing
from tqdm import tqdm # progress bar
from tqdm import trange
from tqdm.contrib.concurrent import process_map
import warnings
import matplotlib
import matplotlib.pyplot as plt

from typing import Union

from abc import ABC, abstractmethod

class FlowPacker(ABC):

    @abstractmethod
    def __init__(self,
                 generator,
                 eps,
                 node_dist,
                 flow_ids,
                 flow_sizes,
                 flow_interarrival_times,
                 network_load_config,
                 auto_node_dist_correction=False,
                 check_dont_exceed_one_ep_load=True,
                 print_data=False,
                 machine_eps=1e-7, # use for avoiding python floating point errors
                 **kwargs,
                 ):
        self.generator = generator
        self.eps = eps
        self.node_dist = copy.deepcopy(node_dist)
        self.flow_ids = flow_ids
        self.flow_sizes = flow_sizes
        self.flow_interarrival_times = flow_interarrival_times
        self.network_load_config = network_load_config
        self.auto_node_dist_correction = auto_node_dist_correction
        self.check_dont_exceed_one_ep_load = check_dont_exceed_one_ep_load
        self.print_data = print_data
        self.machine_eps = machine_eps

        # init useful params
        self.num_nodes, self.num_pairs, self.node_to_index, self.index_to_node = tools.get_network_params(self.eps, all_combinations=True)

        # calc overall load rate
        if self.network_load_config['target_load_fraction'] is not None:
            self.load_rate = self.generator._calc_overall_load_rate(self.flow_sizes, self.flow_interarrival_times)
        else:
            # no particular target load specified, just assume max
            self.load_rate = self.network_load_config['network_rate_capacity']
        if self.load_rate > self.network_load_config['network_rate_capacity']:
            raise Exception(f'load_rate ({self.load_rate}) > maximum network_rate_capacity ({self.network_load_config["network_rate_capacity"]})')
        if self.print_data:
            print(f'Overall network load rate: {self.load_rate}')

        num_pairs = (len(self.eps)**2)-len(self.eps)
        if len(flow_sizes) < num_pairs:
            print('WARNING: {} endpoints have {} possible pairs, but packer has only been given {} flows to pack. This will result in sparse packing, which will limit how accurately the packer is able to replicate the target node distribution. If you do not want this, provide the packer with more flows (e.g. by setting min_num_demands to >> number of possible pairs).'.format(len(self.eps), num_pairs, len(flow_sizes)))

        if self.network_load_config['target_load_fraction'] is not None:
            self._check_node_dist_valid_for_this_load()

    @abstractmethod
    def reset(self, *args, **kwargs):
        pass

    @abstractmethod
    def pack_the_flows(self):
        packed_flows = {}
        return packed_flows

    def _check_node_dist_valid_for_this_load(self):
        for idx in self.index_to_node.keys():
            endpoint_target_load_fraction_of_overall_load = sum(self.node_dist[idx, :])
            endpoint_target_load_rate = endpoint_target_load_fraction_of_overall_load * self.load_rate
            if endpoint_target_load_rate > self.network_load_config['ep_link_capacity']:
                # target load rate is invalid
                if not self.auto_node_dist_correction:
                    # user has not enabled TrafPy automatic node distribution correction
                    raise Exception('Your node distribution is invalid for your specified target load. Overall target network load rate: {} info units per unit time. Endpoint {} (node dist idx {}) target fraction of this overall load: {}. Therefore target load rate for this endpoint is {} info units per unit time, which is too high for this end point which has a maximum capacity of {} info units per unit time, therefore your specified node load distribution is invalid. Change your required src-dst target loads in node_dist, or decrease your specified overall load target, or set auto_node_dist_correction=True to make TrafPy automatically correct the node distribution for you by subtracting the excess load of the invalid endpoint and distribution this excess load amongst other nodes (i.e. as your requested network load tends to 1.0, all end point loads will also tend to 1.0).'.format(self.load_rate, self.index_to_node[idx], idx, endpoint_target_load_fraction_of_overall_load, endpoint_target_load_rate, self.network_load_config['ep_link_capacity']))
                else:
                    print('auto_node_dist_correction set to True. Adjusting node distribution to make it valid...')
                    if self.print_data:
                        print('init node dist before correction:\n{}'.format(self.node_dist))
                    self.eps_at_capacity = {ep: False for ep in self.eps}
                    invalid_ep_found = True
                    adjust_start_t = time.time()
                    while invalid_ep_found:
                        invalid_ep_found = self._auto_correct_node_dist()
                    print(f'Adjusted node distribution in {time.time() - adjust_start_t:.3f} s.')
                    break

    def _auto_correct_node_dist(self):
        max_ep_load_frac = self.network_load_config['ep_link_capacity'] / self.load_rate # max fraction of total network load rate that one end point can take
        excess_ep_load_rates = {ep: None for ep in self.eps}
        invalid_ep_found = False 
        for idx in self.index_to_node.keys():
            endpoint_target_load_fraction_of_overall_load = sum(self.node_dist[idx, :])
            endpoint_target_load_rate = round(endpoint_target_load_fraction_of_overall_load * self.load_rate, 6)
            if self.print_data:
                print('ep {} target load rate: {}'.format(idx, endpoint_target_load_rate))
            if endpoint_target_load_rate > self.network_load_config['ep_link_capacity']:
                # too much load rate on this ep
                invalid_ep_found = True
                excess_ep_load_rates[self.index_to_node[idx]] = endpoint_target_load_rate - self.network_load_config['ep_link_capacity']
                self.eps_at_capacity[self.index_to_node[idx]] = True
                # make ep loads equal on all this ep's pairs such that max ep bandwidth requested
                for pair_idx in self.index_to_node.keys():
                    if pair_idx != idx:
                        self.node_dist[idx, pair_idx] = (max_ep_load_frac / ((self.num_nodes-1)))
                        self.node_dist[pair_idx, idx] = (max_ep_load_frac / ((self.num_nodes-1)))
                # spread excess load evenly across other eps not already at capacity
                free_eps = []
                for ep in excess_ep_load_rates.keys():
                    if not self.eps_at_capacity[ep]:
                        free_eps.append(ep)
                if len(free_eps) == 0:
                    raise Exception('No free end points left to spread excess load across.')
                load_rate_to_spread_per_ep = excess_ep_load_rates[self.index_to_node[idx]] / len(free_eps)
                frac_load_rate_to_spread_per_ep = load_rate_to_spread_per_ep / self.load_rate
                frac_load_rate_to_spread_per_ep_pair = frac_load_rate_to_spread_per_ep / (self.num_nodes-1)
                random.shuffle(free_eps) # shuffle so not always spreading load across same eps
                for ep in free_eps:
                    indices = list(self.index_to_node.keys())
                    random.shuffle(indices)
                    for i in indices:
                        if i != self.node_to_index[ep] and not self.eps_at_capacity[self.index_to_node[i]]:
                            self.node_dist[self.node_to_index[ep], i] += (frac_load_rate_to_spread_per_ep_pair)
                            self.node_dist[i, self.node_to_index[ep]] += (frac_load_rate_to_spread_per_ep_pair)

                endpoint_target_load_fraction_of_overall_load = sum(self.node_dist[idx, :])
                endpoint_target_load_rate = endpoint_target_load_fraction_of_overall_load * self.load_rate
                self.reset() # update params
                if self.print_data:
                    print('updated node dist:\n{}'.format(self.node_dist))

        return invalid_ep_found
