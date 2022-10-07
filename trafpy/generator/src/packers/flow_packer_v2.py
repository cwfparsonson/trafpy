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
                 second_pass_pairs_search_mode: Union['shuffle', 'sample', 'deterministic']='sample',
                 second_pass_pairs_search_sample_mode_factor=100,
                 print_data=False):
        '''
        Args:
            second_pass_pairs_search_mode: For the second pass, the packer
                takes a flow and loops through the available src-dst pairs until it
                finds a pair which it can pack the flow into. In order to stop the
                majority of flows being packed into the first few pairs (by index),
                we need to somehow loop through these pairs in a random fashion (if
                we do not, then we will see a fade phenomenon appear in our
                resultant node distribution heat map after packing; set
                second_pass_pairs_search_mode=deterministic to observe this).
                Setting second_pass_pairs_search_mode=shuffle will randomly shuffle
                the pairs at the start of each pass and prevent this fade
                phenomenon. However, np.random.shuffle(pairs) becomes expensive for
                a large number of pairs, and must be called for each flow (of which
                there may be millions). To circumvent this, rather than randomly
                shuffling, we can instead set second_pass_pairs_search_mode=sample
                to randomly sample a pair to try. We then keep randomly sampling
                pairs from an array of available candidate pairs until we find a
                pair which can accommodate the flow. In order to prevent having to
                generate a list of pair indices with each sample, we do not sample
                without replacement, therefore there is a chance that a given pair
                might be needlessly sampled multiple times when searching for a
                pair for a given flow. However, empirically at scale this method is
                overall faster than performing a shuffle on the whole pairs array. 
                You can test this for yourself by switching betwee sample and shuffle
                mode.
        '''
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
        self.second_pass_pairs_search_mode = second_pass_pairs_search_mode
        self.second_pass_pairs_search_sample_mode_factor = second_pass_pairs_search_sample_mode_factor

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

        # calc target load rate of each src-dst pair
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
        # self.pair_current_total_info_dict = {pair: 0 for pair in self.pair_prob_dict.keys()}
        self.pair_current_total_info = np.zeros(len(self.pair_prob_dict))
        # self.pair_current_distance_from_target_info_dict = {pair: self.pair_target_total_info_dict[pair]-self.pair_current_total_info_dict[pair] for pair in self.pair_prob_dict.keys()} self.pair_current_distance_from_target_info = {pair: self.pair_target_total_info_dict[pair]-self.pair_current_total_info_dict[pair] for pair in self.pair_prob_dict.keys()}
        self.pair_current_distance_from_target_info = np.array(list(self.pair_target_total_info_dict.values()) - self.pair_current_total_info)
        self.max_distance_from_target = np.max(list(self.pair_target_total_info_dict.values())) # keep track of this parameter to avoid doing needless first pass when packing

        # calc max total info during simulation per end point and initialise end point total info tracker
        self.max_total_ep_info = self.network_load_config['ep_link_capacity'] * self.duration
        # # double total ep info so can pack src-dst pairs into links
        # self.max_total_ep_info *= 2
        # calc max info can put on src and dst ports (half ep dedicated to each so / 2)
        self.max_total_port_info = self.max_total_ep_info / 2
        self.ep_total_infos = {ep: 0 for ep in self.eps}
        self.src_total_infos = {ep: 0 for ep in self.eps}
        self.dst_total_infos = {ep: 0 for ep in self.eps}

        # self.pairs = np.asarray(list(self.pair_current_distance_from_target_info_dict.keys()))
        self.pairs = np.asarray(list(self.pair_prob_dict.keys()))

        # init mapping of src and dst node ports to each possible pair
        self.src_port_to_pairs = defaultdict(set)
        self.dst_port_to_pairs = defaultdict(set)
        # init mapping of each pair to its string equivalent once at start so do not have to keep making computationally expensive json.loads() for each flow for each pair for each pass
        self.pair_to_json_loads = {}
        self.pair_to_idx = {}
        for pair_idx, pair in enumerate(self.pairs):
            json_loads_pair = json.loads(pair)
            src, dst = json_loads_pair[0], json_loads_pair[1]
            self.src_port_to_pairs[src].add(pair)
            self.dst_port_to_pairs[dst].add(pair)
            self.pair_to_json_loads[pair] = json_loads_pair
            self.pair_to_idx[pair] = pair_idx

        if self.print_data:
            print('pair prob dict:\n{}'.format(self.pair_prob_dict))
            print('pair target load rate dict:\n{}'.format(self.pair_target_load_rate_dict))
            print('pair target total info dict:\n{}'.format(self.pair_target_total_info_dict))
            print('duration: {}'.format(self.duration))
            print('pair prob dict sum: {}'.format(np.sum(list(self.pair_prob_dict.values()))))
            print('pair target load rate sum: {}'.format(np.sum(list(self.pair_target_load_rate_dict.values()))))
            print('pair target total info sum: {}'.format(np.sum(list(self.pair_target_total_info_dict.values()))))
            print('max total ep info: {}'.format(self.max_total_ep_info))
            print('sum of all flow sizes: {}'.format(sum(self.flow_sizes)))

    def _get_first_pass_mask(self, flow, second_pass_mask):
        # first_pass_mask = np.where(list(self.pair_current_distance_from_target_info_dict.values()) - self.packed_flows[flow]['size'] < 0, 0, 1)
        first_pass_mask = np.where(self.pair_current_distance_from_target_info - self.packed_flows[flow]['size'] < 0, 0, 1)
        return np.logical_and(first_pass_mask, second_pass_mask)

    def _update_second_pass_mask(self, chosen_pair, second_pass_mask):
        json_loads_pair = self.pair_to_json_loads[chosen_pair]
        chosen_src, chosen_dst = json_loads_pair[0], json_loads_pair[1]
        if self.check_dont_exceed_one_ep_load:
            # see if this chosen pair can have any more flows packed into it without exceeding 1.0 endpoint load
            # get minimum flow size of remaining flows to be packed
            min_flow_size_remaining = self.flow_sizes[-1] # HACK: Already sorted flow sizes on reset and are packing from largest to smallest, so smallest will be last idx
            if self.src_total_infos[chosen_src] + min_flow_size_remaining > self.max_total_port_info:
                # any pair with this src port cannot have any more flows packed into it, filter its pairs from being included in any future second pass loops
                src_second_pass_mask = np.isin(self.pairs, list(self.src_port_to_pairs[chosen_src]), invert=True).astype(np.int8)
                src_second_pass_mask = np.logical_and(src_second_pass_mask, second_pass_mask)
            else:
                src_second_pass_mask = second_pass_mask
            if self.dst_total_infos[chosen_dst] + min_flow_size_remaining > self.max_total_port_info:
                # any pair with this dst port cannot have any more flows packed into it, filter its pairs from being included in any future second pass loops
                dst_second_pass_mask = np.isin(self.pairs, list(self.dst_port_to_pairs[chosen_dst]), invert=True).astype(np.int8)
                dst_second_pass_mask = np.logical_and(dst_second_pass_mask, second_pass_mask)
            else:
                dst_second_pass_mask = second_pass_mask
            second_pass_mask = np.logical_and(src_second_pass_mask, dst_second_pass_mask)
        return second_pass_mask

    def _get_masked_data(self, data, mask):
        masked_data = np.ma.masked_array(data, mask)
        return masked_data[masked_data.mask].data

    def _perform_first_pass(self, flow, pairs, verbose=False):
        if verbose:
            print(f'Performing first pass for flow {flow} across {len(pairs)} candidate pairs...')
        pairs = self._prepare_pairs_for_packing(pairs, sort_mode='random') # random shuffle to remove node matrix heat map fade phenomenon
        chosen_pair = None
        for pair in pairs:
            if self._check_if_flow_pair_passes_first_pass(flow, pair):
                chosen_pair = pair
                break
        return chosen_pair

    def _check_if_flow_pair_passes_first_pass(self, flow, pair):
        passes = False
        if self.check_dont_exceed_one_ep_load:
            # make sure neither src or dst of pair would exceed their maximum loads
            if not self._check_if_flow_pair_within_max_load(flow, pair):
                pass
            else:
                if self._check_if_flow_pair_within_target_load(flow, pair):
                    passes = True            
        else:
            # do not worry about exceeding 1.0 load
            if self._check_if_flow_pair_within_target_load(flow, pair):
                passes = True
        return passes

    def _check_if_flow_pair_within_target_load(self, flow, pair, verbose=False):
        within_load = False
        json_loads_pair = self.pair_to_json_loads[pair]
        src, dst = json_loads_pair[0], json_loads_pair[1]
        if verbose:
            print(f'FIRST PASS: Checking flow {flow} pair {pair}...')
        # if self.pair_current_distance_from_target_info_dict[pair] - self.packed_flows[flow]['size'] < 0:
        if self.pair_current_distance_from_target_info[self.pair_to_idx[pair]] - self.packed_flows[flow]['size'] < 0:
            if verbose:
                print(f'FAILED: Would exceed pair target load.')
        else:
            within_load = True
            if verbose:
                print(f'PASS')
        return within_load 

    def _perform_second_pass(self, flow, pairs, verbose=False):
        if verbose:
            print(f'Performing second pass for flow {flow} across {len(pairs)} candidate pairs...')
        chosen_pair = None
        if self.second_pass_pairs_search_mode in ['deterministic', 'shuffle']:
            if self.second_pass_pairs_search_mode == 'shuffle':
                # random shuffle candidate pairs to remove node matrix heat map fade phenomenon
                pairs = self._prepare_pairs_for_packing(pairs, sort_mode='random') # random shuffle to remove node matrix heat map fade phenomenon
            else:
                # deterministically loop through candidate pairs in same way each time
                pass
            for pair in pairs:
                if self._check_if_flow_pair_within_max_load(flow, pair):
                    chosen_pair = pair
                    break
        elif self.second_pass_pairs_search_mode == 'sample':
            # randomly sample a candidate pair until find a pair which passes
            counter = 1
            while counter < len(pairs) * self.second_pass_pairs_search_sample_mode_factor:
                pair = np.random.choice(pairs)
                if self._check_if_flow_pair_passes_second_pass(flow, pair):
                    chosen_pair = pair
                    break
                else:
                    counter += 1
            if counter == len(pairs) * self.second_pass_pairs_search_sample_mode_factor:
                raise Exception(f'Was unable to find a valid src-dst pair for flow {flow} after {int(len(pairs) * self.second_pass_pairs_search_sample_mode_factor)} attempts in the second pass with second_pass_pairs_search_mode=sample. This may be because no pair exists which can accommodate this flow without exceeding 1.0 end point load, or because second_pass_pairs_search_sample_mode_factor (self.second_pass_pairs_search_sample_mode_factor) is too low. Try increasing second_pass_pairs_search_sample_mode_factor, or try setting second_pass_pairs_search_mode to shuffle. If this fails, there is likely a bug somehwere with e.g. second_pass_mask masking too many candidate pairs and/or an error in the traffic loads and src-dst pair capacities generated.')
        else:
            raise Exception(f'Unrecognised second_pass_pairs_search_mode {self.second_pass_pairs_search_mode}')


        return chosen_pair

    def _check_if_flow_pair_passes_second_pass(self, flow, pair):
        passes = False
        if self.check_dont_exceed_one_ep_load:
            if self._check_if_flow_pair_within_max_load(flow, pair):
                passes = True
        else:
            passes = True
        return passes

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
            # don't worry about exceeding 1.0 end point load, just allocate to pair furthest from target load
            within_load = True
        return within_load
        
    def _pack_flow_into_chosen_pair(self, flow, chosen_pair):
        # pack flow into this pair
        # self.pair_current_total_info_dict[chosen_pair] = int(self.pair_current_total_info_dict[chosen_pair] + (self.packed_flows[flow]['size']))
        self.pair_current_total_info[self.pair_to_idx[chosen_pair]] = int(self.pair_current_total_info[self.pair_to_idx[chosen_pair]] + (self.packed_flows[flow]['size']))
        # self.pair_current_distance_from_target_info_dict[chosen_pair] = int(self.pair_current_distance_from_target_info_dict[chosen_pair] - (self.packed_flows[flow]['size']))
        self.pair_current_distance_from_target_info[self.pair_to_idx[chosen_pair]] = int(self.pair_current_distance_from_target_info[self.pair_to_idx[chosen_pair]] - (self.packed_flows[flow]['size']))

        # update max distance from target load across all pairs if necessary
        # if self.pair_current_distance_from_target_info_dict[chosen_pair] > self.max_distance_from_target:
            # self.max_distance_from_target = self.pair_current_distance_from_target_info_dict[chosen_pair]
        if self.pair_current_distance_from_target_info[self.pair_to_idx[chosen_pair]] > self.max_distance_from_target:
            self.max_distance_from_target = self.pair_current_distance_from_target_info[self.pair_to_idx[chosen_pair]]

        # updated packed flows dict
        pair = json.loads(chosen_pair)
        src, dst = pair[0], pair[1]
        self.packed_flows[flow]['src'], self.packed_flows[flow]['dst'] = src, dst 
        self.ep_total_infos[src] += self.packed_flows[flow]['size']
        self.ep_total_infos[dst] += self.packed_flows[flow]['size']
        self.src_total_infos[src] += self.packed_flows[flow]['size']
        self.dst_total_infos[dst] += self.packed_flows[flow]['size']

    def _prepare_pairs_for_packing(self, pairs, sort_mode='random'):
        if sort_mode == 'descending':
            # sort in descending order of total infos
            # sorted_indices = np.argsort(list(self.pair_current_distance_from_target_info_dict.values()))[::-1]
            # sorted_indices = np.argsort(list(self.pair_current_total_info_dict.values()))[::-1]
            sorted_indices = np.argsort(self.pair_current_total_info)[::-1]
            sorted_pairs = pairs[sorted_indices]
        if sort_mode == 'ascending':
            # sort in ascending order of total infos
            # sorted_indices = np.argsort(list(self.pair_current_distance_from_target_info_dict.values()))
            # sorted_indices = np.argsort(list(self.pair_current_total_info_dict.values()))
            sorted_indices = np.argsort(self.pair_current_total_info)
            sorted_pairs = [pairs[sorted_indices]]
        elif sort_mode == 'random':
            # randomly shuffle pair order to prevent unwanted fade trends in node dist
            # sorted_pairs = copy.copy(pairs)
            np.random.shuffle(pairs)
            sorted_pairs = pairs
            # np.random.permutation(sorted_pairs)
        else:
            raise Exception(f'Unrecognised sort_mode {sort_mode}')
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

        '''
        pbar = tqdm(total=len(self.packed_flows.keys()), 
                    desc='Packing flows',
                    leave=False,
                    smoothing=0)
        start = time.time()

        # initialise the second pass mask
        second_pass_mask = np.ones(len(self.pairs), dtype=np.int8)

        # pack each flow into a src-dst pair
        for flow_idx, flow in enumerate(self.packed_flows.keys()):
            chosen_pair = None

            

            # candidate_pairs = self._get_masked_data(data=self.pairs, mask=second_pass_mask)
            # candidate_pair_distances = self._get_masked_data(data=self.pair_current_distance_from_target_info, mask=second_pass_mask)
            # sorted_candidate_indices = np.argsort(candidate_pair_distances)[::-1]
            # sorted_candidate_pairs = candidate_pairs[sorted_candidate_indices]
            # chosen_pair = sorted_candidate_pairs[0]
            


            # first pass (try not to exceed target pair load)
            if self.max_distance_from_target - self.packed_flows[flow]['size'] < 0:
                # there is no way to pack this flow into any pair with the forward pass, can skip straight to second pass
                pass
            else:
                # mask out any pairs which do not meet the first pass requirements for this flow
                first_pass_mask = self._get_first_pass_mask(flow, second_pass_mask)
                first_pass_pairs = self._get_masked_data(data=self.pairs, mask=first_pass_mask)
                # print(f'Performing first pass for flow {flow} across {len(first_pass_pairs)} candidate pairs...') # DEBUG
                chosen_pair = self._perform_first_pass(flow, first_pass_pairs)

            # second pass (if cannot avoid exceeding any pair's target load, pack into pair without exceeding max total load)
            if chosen_pair is None:
                # first pass failed, perform second pass
                second_pass_pairs = self._get_masked_data(data=self.pairs, mask=second_pass_mask)
                # print(f'Performing second pass for flow {flow} across {len(_pairs)} candidate pairs...') # DEBUG
                chosen_pair = self._perform_second_pass(flow, second_pass_pairs)
            else:
                pass

            # check for errors
            if chosen_pair is None:
                # could not find end point pair with enough capacity to take flow
                raise Exception(f'Unable to find valid pair to assign flow {flow}: {self.packed_flows[flow]} without exceeding ep total information load limit {self.max_total_ep_info} information units for this session. Increase number of flows to increase time duration the flow packer has to pack flows into (recommended), and/or decrease flow sizes to help with packing (recommended), and/or increase end point link capacity (recommended), and/or decrease your required target load to increase the time duration the flow packer has to pack flows into, and/or change your node dist to be less heavily skewed. Alternatively, try re-running dist and flow generator since may have chance of creating valid dists and flows which can be packed (also recommended). You can also disable this validity checker by setting check_dont_exceed_one_ep_load to False. Doing so will allow end point loads to go above 1.0 when packing the flows and disable this exception being raised. Alternatively, there may be a bug in the code resulting in this error. Current end point total information loads (information units):\n{self.ep_total_infos}')
            if not self._check_if_flow_pair_within_max_load(flow, chosen_pair):
                # raise Exception(f'ERROR: Flow {flow} with size {self.packed_flows[flow]["size"]} has been allocated to chosen_pair {chosen_pair} which has src total info ({self.src_total_infos[chosen_src]}) and/or dst total info ({self.dst_total_infos[chosen_dst]}) + flow size > max_total_port_info ({self.max_total_port_info}) (pair_current_distance_from_target_info_dict: {self.pair_current_distance_from_target_info_dict[chosen_pair]})')
                json_loads_pair = self.pair_to_json_loads[chosen_pair]
                chosen_src, chosen_dst = json_loads_pair[0], json_loads_pair[1]
                raise Exception(f'ERROR: Flow {flow} with size {self.packed_flows[flow]["size"]} has been allocated to chosen_pair {chosen_pair} which has src total info ({self.src_total_infos[chosen_src]}) and/or dst total info ({self.dst_total_infos[chosen_dst]}) + flow size > max_total_port_info ({self.max_total_port_info}) (pair_current_distance_from_target_info: {self.pair_current_distance_from_target_info[self.pair_to_idx[chosen_pair]]})')

            # pack flow into the chosen src-dst pair
            self._pack_flow_into_chosen_pair(flow, chosen_pair)

            # update second pass mask if necessary
            second_pass_mask = self._update_second_pass_mask(chosen_pair, second_pass_mask)

            pbar.update(1)

        # shuffle flow order to maintain randomness for arrival time in simulation (since sorted flows by size above)
        shuffled_packed_flows = self._shuffle_packed_flows()

        pbar.close()
        end = time.time()
        print('Packed {} flows in {} s.'.format(len(self.packed_flows.keys()), end-start))

        if self.print_data:
            # print('\nFinal total infos at each pair:\n{}'.format(self.pair_current_total_info_dict))
            print('\nFinal total infos at each pair:\n{}'.format(self.pair_current_total_info))
            print('Final total infos at each ep:\n{}'.format(self.ep_total_infos))
            ep_load_rates = {ep: self.ep_total_infos[ep]/self.duration for ep in self.ep_total_infos.keys()}
            print('Corresponding final load rates at each ep:\n{}'.format(ep_load_rates))

        return shuffled_packed_flows
