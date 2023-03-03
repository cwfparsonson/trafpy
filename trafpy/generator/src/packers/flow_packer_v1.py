from trafpy.generator.src.packers.flow_packer import FlowPacker
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


class FlowPackerV1(FlowPacker):
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
                 print_data=False):
        FlowPacker.__init__(
                    self,
                    generator=generator,
                    eps=eps,
                    node_dist=copy.deepcopy(node_dist),
                    flow_ids=flow_ids,
                    flow_sizes=flow_sizes,
                    flow_interarrival_times=flow_interarrival_times,
                    network_load_config=network_load_config,
                    auto_node_dist_correction=auto_node_dist_correction,
                    check_dont_exceed_one_ep_load=check_dont_exceed_one_ep_load,
                    print_data=print_data,
                )

    def reset(self):
        # init dict in which flows will be packed into src-dst pairs
        nested_dict = lambda: defaultdict(nested_dict)
        self.packed_flows = nested_dict()

        # want to pack largest flows first -> re-organise flows into descending order (will shuffle later so maintain random flow sizes of arrivals)
        self.flow_sizes[::-1].sort()
        for idx in range(len(self.flow_ids)):
            self.packed_flows[self.flow_ids[idx]] = {'size': self.flow_sizes[idx],
                                                     'src': None,
                                                     'dst': None}

        # calc overall load rate
        if self.network_load_config['target_load_fraction'] is not None:
            self.load_rate = self.generator._calc_overall_load_rate(self.flow_sizes, self.flow_interarrival_times)
        else:
            # no particular target load specified, just assume max
            self.load_rate = self.network_load_config['network_rate_capacity']
        if self.print_data:
            print('Overall network load rate: {}'.format(self.load_rate))

        # calc target load rate of each src-dst pair
        self.num_nodes, self.num_pairs, self.node_to_index, self.index_to_node = tools.get_network_params(self.eps, all_combinations=True)
        self.pair_prob_dict = node_dists.get_pair_prob_dict_of_node_dist_matrix(self.node_dist, self.eps, all_combinations=True) # N.B. These values sum to 0.5 -> need to allocate twice (src-dst and dst-src)
        self.pair_target_load_rate_dict = {pair: frac*self.load_rate for pair, frac in self.pair_prob_dict.items()}

        # calc target total info to pack into each src-dst pair
        flow_event_times = tools.gen_event_times(self.flow_interarrival_times)
        self.duration = max(flow_event_times) - min(flow_event_times)
        if self.duration == 0:
            # set to some number to prevent infinities
            self.duration = 1e6
        self.pair_target_total_info_dict = {pair: load_rate*self.duration for pair, load_rate in self.pair_target_load_rate_dict.items()}

        # init current total info packed into each src-dst pair and current distance from target info
        self.pair_current_total_info_dict = {pair: 0 for pair in self.pair_prob_dict.keys()}
        self.pair_current_distance_from_target_info_dict = {pair: self.pair_target_total_info_dict[pair]-self.pair_current_total_info_dict[pair] for pair in self.pair_prob_dict.keys()}

        # calc max total info during simulation per end point and initialise end point total info tracker
        self.max_total_ep_info = self.network_load_config['ep_link_capacity'] * self.duration
        # # double total ep info so can pack src-dst pairs into links
        # self.max_total_ep_info *= 2
        # calc max info can put on src and dst ports (half ep dedicated to each so / 2)
        self.max_total_port_info = self.max_total_ep_info / 2
        self.ep_total_infos = {ep: 0 for ep in self.eps}
        self.src_total_infos = {ep: 0 for ep in self.eps}
        self.dst_total_infos = {ep: 0 for ep in self.eps}

        flow_sizes = [self.packed_flows[flow]['size'] for flow in self.packed_flows.keys()]
        if self.print_data:
            print('pair prob dict:\n{}'.format(self.pair_prob_dict))
            print('pair target load rate dict:\n{}'.format(self.pair_target_load_rate_dict))
            print('pair target total info dict:\n{}'.format(self.pair_target_total_info_dict))
            print('duration: {}'.format(self.duration))
            print('pair prob dict sum: {}'.format(np.sum(list(self.pair_prob_dict.values()))))
            print('pair target load rate sum: {}'.format(np.sum(list(self.pair_target_load_rate_dict.values()))))
            print('pair target total info sum: {}'.format(np.sum(list(self.pair_target_total_info_dict.values()))))
            print('max total ep info: {}'.format(self.max_total_ep_info))
            print('sum of all flow sizes: {}'.format(sum(flow_sizes)))

    def _perform_first_pass(self, flow, pairs):
        chosen_pair = None
        for pair in pairs:
            src, dst = json.loads(pair)[0], json.loads(pair)[1]
            if self.check_dont_exceed_one_ep_load:
                # ensure wont exceed 1.0 end point load by allocating this flow to pair
                if self.src_total_infos[src] + self.packed_flows[flow]['size'] > self.max_total_port_info or self.dst_total_infos[dst] + self.packed_flows[flow]['size'] > self.max_total_port_info:
                    # would exceed maximum load for at least one of src and/or dst
                    pass
                else:
                    if self.pair_current_distance_from_target_info_dict[pair] - self.packed_flows[flow]['size'] < 0:
                        # would exceed pair's target total info, try next pair
                        pass
                    else:
                        chosen_pair = pair
                        break
            else:
                # don't worry about exceeding 1.0 end point load, just allocate to pair furthest from target load
                chosen_pair = pair
                break
        return chosen_pair

    def _perform_second_pass(self, flow, pairs):
        chosen_pair = None
        for pair in pairs:
            # pair = pair[0]
            src, dst = json.loads(pair)[0], json.loads(pair)[1]
            if self.check_dont_exceed_one_ep_load:
                # ensure wont exceed 1.0 end point load by allocating this flow to pair
                if self.src_total_infos[src] + self.packed_flows[flow]['size'] > self.max_total_port_info or self.dst_total_infos[dst] + self.packed_flows[flow]['size'] > self.max_total_port_info:
                    # would exceed at least 1 of this pair's end point's maximum load by adding this flow, move to next pair
                    pass
                else:
                    chosen_pair = pair
                    break
            else:
                # don't worry about exceeding 1.0 end point load, just allocate to pair furthest from target load
                chosen_pair = pair
                break
        return chosen_pair
        
    def _pack_flow_into_chosen_pair(self, flow, chosen_pair):
        # pack flow into this pair
        self.pair_current_total_info_dict[chosen_pair] = int(self.pair_current_total_info_dict[chosen_pair] + (self.packed_flows[flow]['size']))
        self.pair_current_distance_from_target_info_dict[chosen_pair] = int(self.pair_current_distance_from_target_info_dict[chosen_pair] - (self.packed_flows[flow]['size']))

        # updated packed flows dict
        pair = json.loads(chosen_pair)
        src, dst = pair[0], pair[1]
        self.packed_flows[flow]['src'], self.packed_flows[flow]['dst'] = src, dst 
        self.ep_total_infos[src] += self.packed_flows[flow]['size']
        self.ep_total_infos[dst] += self.packed_flows[flow]['size']
        self.src_total_infos[src] += self.packed_flows[flow]['size']
        self.dst_total_infos[dst] += self.packed_flows[flow]['size']

    def _prepare_pairs_for_packing_a_flow(self, pairs):
        # randomly shuffle pair order to prevent unwanted fade trends in node dist
        np.random.shuffle(pairs)
        # get pair distances
        distances = np.asarray([self.pair_current_distance_from_target_info_dict[pair] for pair in pairs])
        # sort in descending order
        sorted_indices = np.argsort(distances)[::-1]
        sorted_pairs = pairs[sorted_indices]
        return sorted_pairs

    def _shuffle_packed_flows(self):
        shuffled_packed_flows = {}
        shuffled_keys = list(self.packed_flows.keys())
        random.shuffle(shuffled_keys)
        for shuffled_key in shuffled_keys:
            shuffled_packed_flows[shuffled_key] = self.packed_flows[shuffled_key]
        return shuffled_packed_flows

    def pack_the_flows(self):
        '''
        If you find that your achieved node distribution does not look like
        your original node distribution before packing (e.g. achieved is more
        uniform), is probably because your flow sizes are very large for
        the end point bandwidth you have specified. Either decrease
        your flow sizes or increase the endpoint link capacity
        to make packing easier.

        N.B. If you want to decrease the resultant Jensen Shannon distribution
        of the node distribution from the target node distribution, you can 
        decrease the flow sizes to increase the packing resolution and increase
        the min_num_demands (i.e. the minimum number of flows) to pack. Note that
        as the target load rate scales to 1.0, if the target node distribution
        has not been originally shaped to be compatible with a ~1.0 network load,
        then the resultant node distribution will obviously have a high Jensen
        Shannon distance because the target node distribution had to be auto-adjusted
        to be valid. You can set auto_node_dist_correction=False to stop this behaviour.

        '''
        pbar = tqdm(total=len(self.packed_flows.keys()), 
                    desc='Packing flows',
                    leave=False,
                    smoothing=0)
        packing_start_t = time.time()

        pairs = np.asarray(list(self.pair_current_distance_from_target_info_dict.keys()))
        for flow in self.packed_flows.keys():
            if self.print_data:
                print('\nPacking flow {} of size {}'.format(flow, self.packed_flows[flow]['size']))

            pairs = self._prepare_pairs_for_packing_a_flow(pairs)

            if self.print_data:
                print('Current distance from target info:\n{}'.format(pairs))
                print('Looking for pair furthest from target info...')

            # first pass (try not to exceed target pair load)
            chosen_pair = self._perform_first_pass(flow, pairs)

            if chosen_pair is None:
                # second pass (if can't avoid exceeding any pair's target load, pack into pair without exceeding max total load)
                chosen_pair = self._perform_second_pass(flow, pairs)

            if chosen_pair is None:
                # could not find end point pair with enough capacity to take flow
                raise Exception('Unable to find valid pair to assign flow {}: {} without exceeding ep total information load limit {} information units for this session. Increase number of flows to increase time duration the flow packer has to pack flows into (recommended), and/or decrease flow sizes to help with packing (recommended), and/or increase end point link capacity (recommended), and/or decrease your required target load to increase the time duration the flow packer has to pack flows into, and/or change your node dist to be less heavily skewed. Alternatively, try re-running dist and flow generator since may have chance of creating valid dists and flows which can be packed (also recommended). You can also disable this validity checker by setting check_dont_exceed_one_ep_load to False. Doing so will allow end point loads to go above 1.0 when packing the flows and disable this exception being raised. Current end point total information loads (information units):\n{}\nPair info distances from targets:\n{}'.format(flow, self.packed_flows[flow], self.max_total_ep_info, self.ep_total_infos, self.pair_current_distance_from_target_info_dict))
            
            if self.print_data:
                print('Assigning flow to pair {}'.format(chosen_pair))

            # pack flow into this pair
            self._pack_flow_into_chosen_pair(flow, chosen_pair)

            # # DEBUG
            # if self.pair_current_distance_from_target_info_dict[chosen_pair] < 0:
                # raise Exception()

            pbar.update(1)

        # shuffle flow order to maintain randomness
        shuffled_packed_flows = self._shuffle_packed_flows()

        pbar.close()
        # compute tracker metrics
        self.packing_time = time.time() - packing_start_t
        target_pair_dist = np.array(list(self.pair_prob_dict.values()))
        achieved_pair_load = np.array(list(self.pair_current_total_info_dict.values())) / self.duration
        achieved_pair_dist = achieved_pair_load / np.sum(achieved_pair_load)
        self.packing_jensen_shannon_distance = tools.compute_jensen_shannon_distance(target_pair_dist, achieved_pair_dist)

        print(f'Packed {len(self.packed_flows)} flows in {self.packing_time:.3f} s | Node distribution Jensen Shannon distance from target achieved: {self.packing_jensen_shannon_distance}')

        if self.print_data:
            print('\nFinal total infos at each pair:\n{}'.format(self.pair_current_total_info_dict))
            print('Final total infos at each ep:\n{}'.format(self.ep_total_infos))
            ep_load_rates = {ep: self.ep_total_infos[ep]/self.duration for ep in self.ep_total_infos.keys()}
            print('Corresponding final load rates at each ep:\n{}'.format(ep_load_rates))

        return shuffled_packed_flows
