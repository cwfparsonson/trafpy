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

class FlowPackerV2(FlowPacker):
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
        self.packed_flows = {}

        # want to pack largest flows first -> re-organise flows into descending order (will shuffle later so maintain random flow sizes of arrivals)
        self.flow_sizes[::-1].sort()

        # init packed flows with src-dst pairs to be determined
        for idx in range(len(self.flow_ids)):
            self.packed_flows[self.flow_ids[idx]] = {'size': self.flow_sizes[idx],
                                                     'src': None,
                                                     'dst': None}

        # get each possible src-dst pair and calc their corresponding target load rate
        self.pair_prob_dict = node_dists.get_pair_prob_dict_of_node_dist_matrix(self.node_dist, self.eps, all_combinations=True) # N.B. These values sum to 0.5 -> need to allocate twice (src-dst and dst-src)
        self.pairs = np.asarray(list(self.pair_prob_dict.keys()))
        self.pair_probs = np.asarray(list(self.pair_prob_dict.values()))
        # if np.sum(self.pair_probs) == 1:
        if np.sum(self.pair_probs) + self.machine_eps >= 1:
            # need load fracs to sum to 0.5 since allocate twice (src-dst and dst-src)
            self.pair_probs /= 2
        self.pair_target_load_rate = self.pair_probs * self.load_rate

        # calc target total info to pack into each src-dst pair throughout simulation
        flow_event_times = tools.gen_event_times(self.flow_interarrival_times)
        self.duration = max(flow_event_times) - min(flow_event_times)
        if self.duration == 0:
            # set to some number to prevent infinities
            self.duration = 1e6
        self.pair_target_total_info = self.pair_target_load_rate * self.duration

        # init current total info packed into each src-dst pair and current distance from target info
        self.pair_current_total_info = np.zeros(len(self.pair_prob_dict))
        self.pair_current_distance_from_target_info = self.pair_target_total_info - self.pair_current_total_info

        # calc max total info during simulation per end point and initialise end point total info tracker
        self.max_total_ep_info = self.network_load_config['ep_link_capacity'] * self.duration
        self.max_total_port_info = self.max_total_ep_info / 2 # each end point is split into a src and dst
        self.ep_total_infos = {ep: 0 for ep in self.eps}

        # calc max info can put on src and dst ports
        self.src_total_infos = {ep: 0 for ep in self.eps}
        self.dst_total_infos = {ep: 0 for ep in self.eps}

        # init mapping of src and dst node ports to each possible pair
        self.src_port_to_pairs = defaultdict(set)
        self.dst_port_to_pairs = defaultdict(set)
        # init mapping of each pair to its string equivalent once at start so do not have to keep making computationally expensive json.loads() for each flow for each pair for each pass
        self.pair_to_json_loads = {}
        # init mapping of each pair to its remaining info capacity (the min remaining info capacity of its src-dst)
        self.pair_to_remaining_capacity = []
        # init mapping of each pair to its idx for accessing numpy arrays
        self.pair_to_idx = {}
        for pair_idx, pair in enumerate(self.pairs):
            json_loads_pair = json.loads(pair)
            src, dst = json_loads_pair[0], json_loads_pair[1]
            self.src_port_to_pairs[src].add(pair)
            self.dst_port_to_pairs[dst].add(pair)
            self.pair_to_json_loads[pair] = json_loads_pair
            self.pair_to_remaining_capacity.append(self.max_total_port_info)
            self.pair_to_idx[pair] = pair_idx
        self.pair_to_remaining_capacity = np.array(self.pair_to_remaining_capacity)

        if self.print_data:
            print('Duration: {}'.format(self.duration))
            print('Pair prob sum: {}'.format(np.sum(self.pair_probs)))
            print('Pair target load rate sum: {}'.format(np.sum(self.pair_target_load_rate)))
            print('Pair target total info sum: {}'.format(np.sum(self.pair_target_total_info)))
            print('Max total ep info: {}'.format(self.max_total_ep_info))
            print('Sum of all flow sizes: {}'.format(np.sum(self.flow_sizes)))

    def _get_masked_data(self, data, mask):
        masked_data = np.ma.masked_array(data, mask)
        return masked_data[masked_data.mask].data

    def _check_if_flow_pair_within_max_load(self, flow, pair):
        within_load = False
        json_loads_pair = self.pair_to_json_loads[pair]
        src, dst = json_loads_pair[0], json_loads_pair[1]
        if self.check_dont_exceed_one_ep_load:
            # ensure wont exceed 1.0 end point load by allocating this flow to pair
            if self.src_total_infos[src] + self.packed_flows[flow]['size'] > self.max_total_port_info or self.dst_total_infos[dst] + self.packed_flows[flow]['size'] > self.max_total_port_info:
                # would exceed at least 1 of this pair's end point's maximum load by adding this flow, move to next pair
                pass
            else:
                within_load = True
        else:
            # do not worry about exceeding 1.0 end point load, just allocate to pair furthest from target load
            within_load = True
        return within_load
        
    def _pack_flow_into_chosen_pair(self, flow, chosen_pair):
        # pack flow into this pair
        self.pair_current_total_info[self.pair_to_idx[chosen_pair]] = self.pair_current_total_info[self.pair_to_idx[chosen_pair]] + (self.packed_flows[flow]['size'])
        self.pair_current_distance_from_target_info[self.pair_to_idx[chosen_pair]] = self.pair_current_distance_from_target_info[self.pair_to_idx[chosen_pair]] - (self.packed_flows[flow]['size'])

        # updated packed flows dict
        json_loads_pair = self.pair_to_json_loads[chosen_pair]
        chosen_src, chosen_dst = json_loads_pair[0], json_loads_pair[1]
        self.packed_flows[flow]['src'], self.packed_flows[flow]['dst'] = chosen_src, chosen_dst 

        # update end point and src-dst port info of chosen pair
        self.ep_total_infos[chosen_src] += self.packed_flows[flow]['size']
        self.ep_total_infos[chosen_dst] += self.packed_flows[flow]['size']
        self.src_total_infos[chosen_src] += self.packed_flows[flow]['size']
        self.dst_total_infos[chosen_dst] += self.packed_flows[flow]['size']

        # update src-dst info of any other pairs associated with this chosen pair's src and dst
        for pair in self.src_port_to_pairs[chosen_src]:
            json_loads_pair = self.pair_to_json_loads[pair]
            dst = json_loads_pair[1]
            self.pair_to_remaining_capacity[self.pair_to_idx[pair]] = min(self.max_total_port_info - self.src_total_infos[chosen_src], self.max_total_port_info - self.dst_total_infos[dst])
        for pair in self.dst_port_to_pairs[chosen_dst]:
            json_loads_pair = self.pair_to_json_loads[pair]
            src = json_loads_pair[0]
            self.pair_to_remaining_capacity[self.pair_to_idx[pair]] = min(self.max_total_port_info - self.src_total_infos[src], self.max_total_port_info - self.dst_total_infos[chosen_dst])

    def _shuffle_packed_flows(self):
        shuffled_packed_flows = {}
        shuffled_keys = list(self.packed_flows.keys())
        random.shuffle(shuffled_keys)
        for shuffled_key in shuffled_keys:
            shuffled_packed_flows[shuffled_key] = self.packed_flows[shuffled_key]
        return shuffled_packed_flows

    def _choose_pair(self, flow):
        if self.check_dont_exceed_one_ep_load:
            # mask out pairs whose src and/or dst would exceed 1.0 load rate were they to be allocated this flow
            pairs_mask = np.where(self.pair_to_remaining_capacity - self.packed_flows[flow]['size'] < 0, 0, 1)
            candidate_pairs = self._get_masked_data(data=self.pairs, mask=pairs_mask)
            # get the candidate pair distances adjusted for their total target information, as this will determine packing priority to accurately reproduce the distribution
            adjusted_candidate_pair_distances = self._get_masked_data(data=self.pair_current_distance_from_target_info + self.pair_target_total_info, mask=pairs_mask)
        else:
            # no need to worry about exceeding 1.0 load rate
            candidate_pairs = self.pairs
            adjusted_candidate_pair_distances = self.pair_current_distance_from_target_info + self.pair_target_total_info

        # find the indices of the valid pairs which are furthest from their target load and therefore should have a flow packed into them
        max_indices = np.argwhere(adjusted_candidate_pair_distances == np.amax(adjusted_candidate_pair_distances)).flatten()

        # randomly select a pair to avoid fade phenomenon in the resultant node dist
        return candidate_pairs[np.random.choice(max_indices)]

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

        # pack each flow into a src-dst pair
        for flow in self.packed_flows.keys():

            # choose a src-dst pair to pack this flow into
            chosen_pair = self._choose_pair(flow)

            if self.check_dont_exceed_one_ep_load:
                if not self._check_if_flow_pair_within_max_load(flow, chosen_pair):
                    json_loads_pair = self.pair_to_json_loads[chosen_pair]
                    chosen_src, chosen_dst = json_loads_pair[0], json_loads_pair[1]
                    raise Exception(f'ERROR: Flow {flow} with size {self.packed_flows[flow]["size"]} has been allocated to chosen_pair {chosen_pair} which has src total info ({self.src_total_infos[chosen_src]}) and/or dst total info ({self.dst_total_infos[chosen_dst]}) + flow size > max_total_port_info ({self.max_total_port_info}) (pair_current_distance_from_target_info: {self.pair_current_distance_from_target_info[self.pair_to_idx[chosen_pair]]} | pair_to_remaining_capacity: {self.pair_to_remaining_capacity[self.pair_to_idx[chosen_pair]]})')

            # pack flow into the chosen src-dst pair
            self._pack_flow_into_chosen_pair(flow, chosen_pair)

            pbar.update(1)

        # shuffle flow order to maintain randomness for arrival time in simulation (since sorted flows by size above)
        shuffled_packed_flows = self._shuffle_packed_flows()

        pbar.close()

        # compute tracker metrics
        self.packing_time = time.time() - packing_start_t
        if np.sum(self.pair_probs) - self.machine_eps <= 0.5:
            target_pair_dist = self.pair_probs * 2 # sum to 1.0 (previously summed to 0.5 since allocated twice for src-dst and dst-src)
        else:
            target_pair_dist = self.pair_probs
        achieved_pair_load = self.pair_current_total_info / self.duration
        achieved_pair_dist = achieved_pair_load / np.sum(achieved_pair_load)
        self.packing_jensen_shannon_distance = tools.compute_jensen_shannon_distance(target_pair_dist, achieved_pair_dist)

        print(f'Packed {len(self.packed_flows)} flows in {self.packing_time:.3f} s | Node distribution Jensen Shannon distance from target achieved: {self.packing_jensen_shannon_distance}')

        return shuffled_packed_flows
