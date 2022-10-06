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

# import ray
# import psutil
# NUM_CPUS = psutil.cpu_count(logical=False)
# try:
    # ray.init(num_cpus=NUM_CPUS)
# except RuntimeError:
    # # already initialised ray in another script, no need to init again
    # pass




class FlowGenerator:
    def __init__(self,
                 eps,
                 node_dist,
                 flow_size_dist,
                 interarrival_time_dist,
                 network_load_config,
                 min_num_demands=6000,
                 max_num_demands=None,
                 jensen_shannon_distance_threshold=0.1,
                 min_last_demand_arrival_time=None,
                 auto_node_dist_correction=False,
                 check_dont_exceed_one_ep_load=True,
                 print_data=False):
        '''
        Args:
            network_load_config (dict): Dict of form {'network_rate_capacity': <int/float>, 'target_load_fraction': <float>, 'disable_timeouts': <bool>, 'return_new_interarrival_time_dist': <bool>},
                where network_rate_capacity is the maximum rate (in e.g. Gbps) at which
                information can be reliably transmitted over the communication network
                which the demand data will be inserted into, and where target_load_fraction
                is the fraction of the network rate capacity being requested by the demands
                (e.g. target_load_fraction=0.75 would generate demands which request
                a load that is 75% of the network rate capacity from the first to 
                the last demand arriving). If 'target_load_fraction' is None, won't adjust
                inter arrival time dist at all to meet network load.
            auto_node_dist_correction (bool): Set to True if you want TrafPy to
                automatically make invalid node distributions valid. If True, invalid
                node distributions where more load is being assigned to a end point
                link than the end point link has bandwidth will be changed by 
                removing the invalid end point link load to its maximum 1.0 load
                and distributing the removed load across all other valid links
                uniformly.
            max_num_demands (int): If not None, will not exceed this number of demands,
                which can help if you find you are exceeding memory limitations in
                your simulations. However, this will also mean that the (1) jensen_shannon_distance_threshold
                and (2) min_last_demand_arrival_time you specifiy may not be met. To
                ensure these are met, you must set max_num_demands to None.
            jensen_shannon_distance_threshold (float): Maximum jensen shannon distance
                required of generated random variables w.r.t. discretised dist they're generated from.
                Must be between 0 and 1. Distance of 0 -> distributions are exactly the same.
                Distance of 1 -> distributions are not at all similar.
                https://medium.com/datalab-log/measuring-the-statistical-similarity-between-two-samples-using-jensen-shannon-and-kullback-leibler-8d05af514b15
                N.B. To meet threshold, this function will keep doubling num_demands
            check_dont_exceed_one_ep_load (bool): If True, when packing flows (assigning
                src-dst node pairs according to specified node distribution), will ensure
                that don't exceed 1.0 load on any end points. If this is not possible,
                will raise an Exception. If False, no exception will be raised, but run
                risk of exceeding 1.0 end point load, which for some users might be
                detrimental to their system.

        '''
        self.started = time.time()

        self.eps = eps
        self.node_dist = node_dist
        self.flow_size_dist = flow_size_dist
        self.interarrival_time_dist = interarrival_time_dist
        self.max_num_demands = max_num_demands
        if min_num_demands is None:
            min_num_demands = 1
        if max_num_demands is not None:
            self.num_demands = min(min_num_demands, max_num_demands)
        else:
            self.num_demands = min_num_demands
        self.network_load_config = network_load_config
        self.min_last_demand_arrival_time = min_last_demand_arrival_time
        self.auto_node_dist_correction = auto_node_dist_correction
        self.jensen_shannon_distance_threshold = jensen_shannon_distance_threshold
        self.check_dont_exceed_one_ep_load = check_dont_exceed_one_ep_load
        self.print_data = print_data

        self.num_nodes, self.num_pairs, self.node_to_index, self.index_to_node = tools.get_network_params(self.eps)

        if self.network_load_config['target_load_fraction'] is not None:
            if self.network_load_config['target_load_fraction'] > 0.95:
                raise Exception('Target load fraction {} is invalid. Must be <= 0.95.'.format(self.network_load_config['target_load_fraction']))

        if not self.check_dont_exceed_one_ep_load:
            print('WARNING: check_dont_exceed_one_ep_load is set to False. This may result in end point loads going above 1.0, which for some users might be detrimental to the systems they want to test.')

    def create_flow_centric_demand_data(self):
        # flow sizes
        flow_sizes = val_dists.gen_rand_vars_from_discretised_dist(unique_vars=list(self.flow_size_dist.keys()),
                                                                   probabilities=list(self.flow_size_dist.values()),
                                                                   num_demands=self.num_demands,
                                                                   jensen_shannon_distance_threshold=self.jensen_shannon_distance_threshold)
        # update num_demands in case jensen-shannon distance threshold required num_demands to be increased
        self.num_demands = max(len(flow_sizes), self.num_demands)

        # flow interarrival times
        interarrival_times = val_dists.gen_rand_vars_from_discretised_dist(unique_vars=list(self.interarrival_time_dist.keys()),
                                                                           probabilities=list(self.interarrival_time_dist.values()),
                                                                           num_demands=self.num_demands,
                                                                           jensen_shannon_distance_threshold=self.jensen_shannon_distance_threshold)
        # update num_demands in case jensen-shannon distance threshold required num_demands to be increased
        self.num_demands = max(len(interarrival_times), self.num_demands)
        if self.num_demands > len(flow_sizes):
            # must sample flow sizes with updated num_demands
            flow_sizes = val_dists.gen_rand_vars_from_discretised_dist(unique_vars=list(self.flow_size_dist.keys()),
                                                                       probabilities=list(self.flow_size_dist.values()),
                                                                       num_demands=self.num_demands,
                                                                       jensen_shannon_distance_threshold=None)
        if self.network_load_config['target_load_fraction'] is not None:
            # adjust overall interarrival time dist until overall load <= user-specified load
            interarrival_times = self._adjust_demand_load(flow_sizes,
                                                          interarrival_times)

        if self.max_num_demands is not None:
            if len(interarrival_times) > self.max_num_demands or len(flow_sizes) > self.max_num_demands:
                print('WARNING: max_num_demands is {} but needed {} flows to meet jensen_shannon_distance_threshold {}. Capping num_demands to max_num_demands, therefore may not meet jensen_shannon_distance_threshold specified. Increase max_num_demands to ensure you meet the jensen_shannon_distance_threshold.'.format(self.max_num_demands, len(flow_sizes), self.jensen_shannon_distance_threshold))
                self.num_demands = self.max_num_demands

                # sample flow sizes and interarrival times with max_num_demands
                flow_sizes = val_dists.gen_rand_vars_from_discretised_dist(unique_vars=list(self.flow_size_dist.keys()),
                                                                           probabilities=list(self.flow_size_dist.values()),
                                                                           num_demands=self.max_num_demands,
                                                                           jensen_shannon_distance_threshold=None)
                interarrival_times = val_dists.gen_rand_vars_from_discretised_dist(unique_vars=list(self.interarrival_time_dist.keys()),
                                                                                   probabilities=list(self.interarrival_time_dist.values()),
                                                                                   num_demands=self.max_num_demands,
                                                                                   jensen_shannon_distance_threshold=self.jensen_shannon_distance_threshold)
                # adjust overall interarrival time dist until overall load <= user-specified load
                interarrival_times = self._adjust_demand_load(flow_sizes,
                                                              interarrival_times)


        # corresponding flow event (arrival) times
        event_times = tools.gen_event_times(interarrival_times)
        index, event_times_sorted = np.argsort(event_times), np.sort(event_times)

        # flow ids
        flow_ids = ['flow_'+str(i) for i in range(self.num_demands)]
        establish = [1 for _ in range(self.num_demands)]

        # pack the flows into src-dst pairs to meet src-dst pair load config requirements of node_dist
        packer = FlowPacker(self,
                            self.eps,
                            self.node_dist,
                            flow_ids,
                            flow_sizes,
                            interarrival_times,
                            network_load_config=self.network_load_config,
                            auto_node_dist_correction=self.auto_node_dist_correction,
                            check_dont_exceed_one_ep_load=self.check_dont_exceed_one_ep_load)
        packed_flows = packer.pack_the_flows()

        # compile packed flows into demand_data dict ordered in terms of arrival time
        demand_data = {'flow_id': [],
                        'sn': [],
                        'dn': [],
                        'flow_size': [],
                        'event_time': event_times_sorted,
                        'establish': establish,
                        'index': index}
        for flow in packed_flows.keys():
            demand_data['flow_id'].append(flow)
            demand_data['sn'].append(packed_flows[flow]['src'])
            demand_data['dn'].append(packed_flows[flow]['dst'])
            demand_data['flow_size'].append(packed_flows[flow]['size'])

        if self.min_last_demand_arrival_time is not None:
            # duplicate flows until get duration >= user-specified duration
            adjustment_factor = self.min_last_demand_arrival_time / max(demand_data['event_time'])
            num_duplications = math.ceil(math.log(adjustment_factor, 2))
            if self.max_num_demands is not None:
                if (2**num_duplications) * len(demand_data['flow_id']) > self.max_num_demands:
                    print('WARING: max_num_demands is {} but have specified min_last_demand_arrival_time {}. Would need {} demands to reach this min_last_demand_arrival_time, therefore must increase max_num_demands (or set to None) if you want to meet this min_last_demand_arrival_time.'.format(self.max_num_demands, self.min_last_demand_arrival_time, (2**num_duplications)*len(demand_data['flow_id'])))
                    return demand_data
            if num_duplications > 0:
                # duplicate
                demand_data = duplicate_demands_in_demand_data_dict(demand_data, 
                                                                    num_duplications=num_duplications,
                                                                    use_multiprocessing=False)

        return demand_data


    def _calc_overall_load_rate(self, flow_sizes, interarrival_times):
        '''Returns load rate (info units per unit time).'''
        info_arrived = self._calc_total_info_arrived(flow_sizes)
        first_flow_arrival_time, last_flow_arrival_time = self._get_first_last_flow_arrival_times(interarrival_times)
        duration = last_flow_arrival_time - first_flow_arrival_time
        return info_arrived/duration

    def _calc_total_info_arrived(self, flow_sizes):
        return np.sum(flow_sizes)

    def _get_first_last_flow_arrival_times(self, interarrival_times):
        event_times = tools.gen_event_times(interarrival_times)
        return min(event_times), max(event_times)

    def _change_interarrival_times_by_factor(self, interarrival_times, factor):
        '''Updates self.interarrival_time_dist by a specified factor and returns new interarrival times.'''
        new_interarrival_time_dist = {}
        for rand_var, prob in self.interarrival_time_dist.items():
            new_rand_var = rand_var * factor
            new_interarrival_time_dist[new_rand_var] = prob

        # update interarrival time dist
        self.interarrival_time_dist = new_interarrival_time_dist

        # gen new interarrival times
        # NEW METHOD: Just scale interarrival times as you have scaled interarrival time dist
        # interarrival_times *= factor
        interarrival_times = (interarrival_times * factor)

        # OLD METHOD: Re-generates by sampling from new dist. CON: Leads to different data -> get different loads.
        # interarrival_times = val_dists.gen_rand_vars_from_discretised_dist(unique_vars=list(self.interarrival_time_dist.keys()),
                                                                           # probabilities=list(self.interarrival_time_dist.values()),
                                                                           # num_demands=self.num_demands)

        return interarrival_times

    def _adjust_demand_load(self,
                            flow_sizes,
                            interarrival_times):
        # total info arriving (sum of flow sizes) is fixed
        # therefore to adjust load, must adjust duration by adjusting interarrival time dist
        load_rate = self._calc_overall_load_rate(flow_sizes, interarrival_times)
        load_fraction = load_rate / self.network_load_config['network_rate_capacity']
        adjustment_factor = load_fraction / self.network_load_config['target_load_fraction'] 
        interarrival_times = self._change_interarrival_times_by_factor(interarrival_times, adjustment_factor)

        return interarrival_times



# @ray.remote
# def _check_if_flow_pair_within_target_load_asynchronous(*args, **kwargs):
    # return _check_if_flow_pair_within_target_load_synchronous(*args, **kwargs)

# def _check_if_flow_pair_within_target_load_synchronous(flow_size,
                                                     # src,
                                                     # src_total_info,
                                                     # dst,
                                                     # dst_total_info,
                                                     # max_total_port_info,
                                                     # pair_current_distance_from_target_info_dict,
                                                     # ):
    # within_load = False
    # src, dst = json.loads(pair)[0], json.loads(pair)[1]
    # if self.check_dont_exceed_one_ep_load:
        # # ensure will not exceed 1.0 end point load by allocating this flow to pair
        # if self.src_total_infos[src] + self.packed_flows[flow]['size'] > self.max_total_port_info or self.dst_total_infos[dst] + self.packed_flows[flow]['size'] > self.max_total_port_info:
            # # would exceed maximum load for at least one of src and/or dst
            # pass
        # else:
            # if self.pair_current_distance_from_target_info_dict[pair] - self.packed_flows[flow]['size'] < 0:
                # # would exceed pair's target total info, try next pair
                # pass
            # else:
                # within_load = True
    # return within_load 




class FlowPacker:
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
        # self.print_data = True # DEBUG

        num_pairs = (len(self.eps)**2)-len(self.eps)
        if len(flow_sizes) < num_pairs:
            print('WARNING: {} endpoints have {} possible pairs, but packer has only been given {} flows to pack. This will result in sparse packing, which will limit how accurately the packer is able to replicate the target node distribution. If you do not want this, provide the packer with more flows (e.g. by setting min_num_demands to >> number of possible pairs).'.format(len(self.eps), num_pairs, len(flow_sizes)))

        self.reset()
        if self.network_load_config['target_load_fraction'] is not None:
            self._check_node_dist_valid_for_this_load()

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
        if self.load_rate > self.network_load_config['network_rate_capacity']:
            raise Exception(f'load_rate ({self.load_rate}) > maximum network_rate_capacity ({self.network_load_config["network_rate_capacity"]})')
        if self.print_data:
            print(f'Overall network load rate: {self.load_rate}')

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

        self.pairs = np.asarray(list(self.pair_current_distance_from_target_info_dict.keys()))

        # init mapping of src and dst node ports to each possible pair
        self.src_port_to_pairs = defaultdict(set)
        self.dst_port_to_pairs = defaultdict(set)
        # init mapping of each pair to its string equivalent once at start so do not have to keep making computationally expensive json.loads() for each flow for each pair for each pass
        self.pair_to_json_loads = {}
        for pair in self.pairs:
            json_loads_pair = json.loads(pair)
            src, dst = json_loads_pair[0], json_loads_pair[1]
            self.src_port_to_pairs[src].add(pair)
            self.dst_port_to_pairs[dst].add(pair)
            self.pair_to_json_loads[pair] = json_loads_pair

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
        first_pass_mask = np.where(list(self.pair_current_distance_from_target_info_dict.values()) - self.packed_flows[flow]['size'] < 0, 0, 1)
        return (first_pass_mask == 1) & (second_pass_mask == 1)

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
                src_second_pass_mask = (src_second_pass_mask == 1) & (second_pass_mask == 1)
            else:
                src_second_pass_mask = second_pass_mask
            if self.dst_total_infos[chosen_dst] + min_flow_size_remaining > self.max_total_port_info:
                # any pair with this dst port cannot have any more flows packed into it, filter its pairs from being included in any future second pass loops
                dst_second_pass_mask = np.isin(self.pairs, list(self.dst_port_to_pairs[chosen_dst]), invert=True).astype(np.int8)
                dst_second_pass_mask = (dst_second_pass_mask == 1) & (second_pass_mask == 1)
            else:
                dst_second_pass_mask = second_pass_mask
            second_pass_mask = (src_second_pass_mask == 1) & (dst_second_pass_mask == 1)
        return second_pass_mask

    def _get_masked_pairs(self, mask):
        masked_pairs = np.ma.masked_array(self.pairs, mask)
        return masked_pairs[masked_pairs.mask].data

    def _perform_first_pass(self, flow, pairs, verbose=False):
        if verbose:
            print(f'Performing first pass for flow {flow} across {len(pairs)} candidate pairs...')
        pairs = self._prepare_pairs_for_packing(pairs, sort_mode='random') # random shuffle to remove node matrix heat map fade phenomenon
        chosen_pair = None
        for pair in pairs:
            if self.check_dont_exceed_one_ep_load:
                # make sure neither src or dst of pair would exceed their maximum loads
                if not self._check_if_flow_pair_within_max_load(flow, pair):
                    pass
                else:
                    if self._check_if_flow_pair_within_target_load(flow, pair):
                        chosen_pair = pair
                        break
            else:
                # do not worry about exceeding 1.0 load
                if self._check_if_flow_pair_within_target_load(flow, pair):
                    chosen_pair = pair
                    break
        return chosen_pair

    def _check_if_flow_pair_within_target_load(self, flow, pair, verbose=False):
        within_load = False
        json_loads_pair = self.pair_to_json_loads[pair]
        src, dst = json_loads_pair[0], json_loads_pair[1]
        if verbose:
            print(f'FIRST PASS: Checking flow {flow} pair {pair}...')
        if self.pair_current_distance_from_target_info_dict[pair] - self.packed_flows[flow]['size'] < 0:
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
        pairs = self._prepare_pairs_for_packing(pairs, sort_mode='random') # random shuffle to remove node matrix heat map fade phenomenon
        chosen_pair = None
        for pair in pairs:
            if self._check_if_flow_pair_within_max_load(flow, pair):
                chosen_pair = pair
                break
        return chosen_pair

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
        self.pair_current_total_info_dict[chosen_pair] = int(self.pair_current_total_info_dict[chosen_pair] + (self.packed_flows[flow]['size']))
        self.pair_current_distance_from_target_info_dict[chosen_pair] = int(self.pair_current_distance_from_target_info_dict[chosen_pair] - (self.packed_flows[flow]['size']))

        # update max distance from target load across all pairs if necessary
        if self.pair_current_distance_from_target_info_dict[chosen_pair] > self.max_distance_from_target:
            self.max_distance_from_target = self.pair_current_distance_from_target_info_dict[chosen_pair]

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
            sorted_indices = np.argsort(list(self.pair_current_total_info_dict.values()))[::-1]
            sorted_pairs = pairs[sorted_indices]
        if sort_mode == 'ascending':
            # sort in ascending order of total infos
            # sorted_indices = np.argsort(list(self.pair_current_distance_from_target_info_dict.values()))
            sorted_indices = np.argsort(list(self.pair_current_total_info_dict.values()))
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

            # first pass (try not to exceed target pair load)
            if self.max_distance_from_target - self.packed_flows[flow]['size'] < 0:
                # there is no way to pack this flow into any pair with the forward pass, can skip straight to second pass
                pass
            else:
                # mask out any pairs which do not meet the first pass requirements for this flow
                first_pass_mask = self._get_first_pass_mask(flow, second_pass_mask)
                first_pass_pairs = self._get_masked_pairs(first_pass_mask)
                # print(f'Performing first pass for flow {flow} across {len(first_pass_pairs)} candidate pairs...') # DEBUG
                chosen_pair = self._perform_first_pass(flow, first_pass_pairs)

            # second pass (if cannot avoid exceeding any pair's target load, pack into pair without exceeding max total load)
            if chosen_pair is None:
                # first pass failed, perform second pass
                second_pass_pairs = self._get_masked_pairs(second_pass_mask)
                # print(f'Performing second pass for flow {flow} across {len(_pairs)} candidate pairs...') # DEBUG
                chosen_pair = self._perform_second_pass(flow, second_pass_pairs)
            else:
                pass

            # check for errors
            if chosen_pair is None:
                # could not find end point pair with enough capacity to take flow
                raise Exception(f'Unable to find valid pair to assign flow {flow}: {self.packed_flows[flow]} without exceeding ep total information load limit {self.max_total_ep_info} information units for this session. Increase number of flows to increase time duration the flow packer has to pack flows into (recommended), and/or decrease flow sizes to help with packing (recommended), and/or increase end point link capacity (recommended), and/or decrease your required target load to increase the time duration the flow packer has to pack flows into, and/or change your node dist to be less heavily skewed. Alternatively, try re-running dist and flow generator since may have chance of creating valid dists and flows which can be packed (also recommended). You can also disable this validity checker by setting check_dont_exceed_one_ep_load to False. Doing so will allow end point loads to go above 1.0 when packing the flows and disable this exception being raised. Alternatively, there may be a bug in the code resulting in this error. Current end point total information loads (information units):\n{self.ep_total_infos}')
            if not self._check_if_flow_pair_within_max_load(flow, chosen_pair):
                raise Exception(f'ERROR: Flow {flow} with size {self.packed_flows[flow]["size"]} has been allocated to chosen_pair {chosen_pair} which has src total info ({self.src_total_infos[chosen_src]}) and/or dst total info ({self.dst_total_infos[chosen_dst]}) + flow size > max_total_port_info ({self.max_total_port_info}) (pair_current_distance_from_target_info_dict: {self.pair_current_distance_from_target_info_dict[chosen_pair]})')

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
            print('\nFinal total infos at each pair:\n{}'.format(self.pair_current_total_info_dict))
            print('Final total infos at each ep:\n{}'.format(self.ep_total_infos))
            ep_load_rates = {ep: self.ep_total_infos[ep]/self.duration for ep in self.ep_total_infos.keys()}
            print('Corresponding final load rates at each ep:\n{}'.format(ep_load_rates))

        return shuffled_packed_flows
            
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
                    self.eps_at_capacity = {ep: False for ep in self.ep_total_infos.keys()}
                    invalid_ep_found = True
                    while invalid_ep_found:
                        invalid_ep_found = self._auto_correct_node_dist()
                    break

    def _auto_correct_node_dist(self):
        max_ep_load_frac = self.network_load_config['ep_link_capacity'] / self.load_rate # max fraction of total network load rate that one end point can take
        excess_ep_load_rates = {ep: None for ep in self.ep_total_infos.keys()}
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
















        

        












# FUNCTIONS

def get_flow_centric_demand_data_ep_load_rate(demand_data, ep, eps, method='all_eps'):
    '''
    If method=='all_eps', duration is time_last_flow_arrived-time_first_flow_arrived
    across all endpoints. If method=='per_ep', duration is time_last_flow_arrived-time_first_flow_arrived
    for this specific ep.
    '''
    ep_info = group_demand_data_into_ep_info(demand_data, eps)
    total_info = sum(ep_info[ep]['flow_size'])
    # if method == 'per_ep':
        # time_first_flow_arrived = min(ep_info[ep]['event_time'])
        # time_last_flow_arrived = max(ep_info[ep]['event_time'])
    # if method == 'all_eps':
    time_first_flow_arrived = min(demand_data['event_time'])
    time_last_flow_arrived = max(demand_data['event_time'])
    duration = time_last_flow_arrived - time_first_flow_arrived
    if duration != 0:
        load_rate = total_info / duration
    else:
        load_rate = float('inf')
    
    return load_rate

def get_flow_centric_demand_data_overall_load_rate(demand_data):
    '''
    If method == 'mean_per_ep', will calculate the total network load as being the mean
    average load on each endpoint link (i.e. sum info requests for each link ->
    find load of each link -> find mean of ep link loads)

    If method == 'mean_all_eps', will calculate the total network load as being
    the average load over all endpoint links (i.e. sum info requests for all links
    -> find overall load of network)
    '''
    info_arrived = get_flow_centric_demand_data_total_info_arrived(demand_data)
    first_event_time, last_event_time = get_first_last_flow_arrival_times(demand_data)
    duration = last_event_time - first_event_time

    if duration != 0:
        load_rate = info_arrived/duration
    else:
        load_rate = float('inf')

    return load_rate



def group_demand_data_into_ep_info(demand_data, eps):
    nested_dict = lambda: defaultdict(nested_dict)
    ep_info = nested_dict()
    added_flow = {flow_id: False for flow_id in demand_data['flow_id']}
    for ep in eps:
        ep_info[ep]['flow_size'] = []
        # ep_info[ep]['event_time'] = []
        ep_info[ep]['demand_data_idx'] = []
        ep_info[ep]['flow_id'] = []
        # ep_info[ep]['establish'] = []
        # ep_info[ep]['index'] = []
        ep_info[ep]['sn'] = []
        ep_info[ep]['dn'] = []
    # group demand data by ep
    for idx in range(len(demand_data['flow_id'])):
        if not added_flow[demand_data['flow_id'][idx]]: 
            # not yet added this flow
            ep_info[demand_data['sn'][idx]]['flow_size'].append(demand_data['flow_size'][idx])
            ep_info[demand_data['dn'][idx]]['flow_size'].append(demand_data['flow_size'][idx])
            # ep_info[demand_data['sn'][idx]]['event_time'].append(demand_data['event_time'][idx])
            # ep_info[demand_data['dn'][idx]]['event_time'].append(demand_data['event_time'][idx])
            ep_info[demand_data['sn'][idx]]['demand_data_idx'].append(idx)
            ep_info[demand_data['dn'][idx]]['demand_data_idx'].append(idx)
            ep_info[demand_data['sn'][idx]]['flow_id'].append(demand_data['flow_id'][idx])
            ep_info[demand_data['dn'][idx]]['flow_id'].append(demand_data['flow_id'][idx])
            # ep_info[demand_data['sn'][idx]]['establish'].append(demand_data['establish'][idx])
            # ep_info[demand_data['dn'][idx]]['establish'].append(demand_data['establish'][idx])
            # ep_info[demand_data['sn'][idx]]['index'].append(demand_data['index'][idx])
            # ep_info[demand_data['dn'][idx]]['index'].append(demand_data['index'][idx])
            ep_info[demand_data['sn'][idx]]['sn'].append(demand_data['sn'][idx])
            ep_info[demand_data['sn'][idx]]['dn'].append(demand_data['dn'][idx])
            ep_info[demand_data['dn'][idx]]['sn'].append(demand_data['sn'][idx])
            ep_info[demand_data['dn'][idx]]['dn'].append(demand_data['dn'][idx])
        else:
            # already added this flow
            pass

    return ep_info


def get_flow_centric_demand_data_total_info_arrived(demand_data): 
    info_arrived = 0
    # print('flow size {} {}'.format(type(demand_data['flow_size']), type(demand_data['flow_size'][0])))
    # if 'job_id' in demand_data:
        # # jobcentric
        # flow_data = demand_data['flow_data']
    # else:
        # # flowcentric
        # flow_data = demand_data

    for flow_size in demand_data['flow_size']:
        if flow_size > 0:
            info_arrived += flow_size
        else:
            pass
    
    return info_arrived
            
def get_first_last_flow_arrival_times(demand_data):
    arrival_times = []
    for idx in range(len(demand_data['event_time'])):
        if demand_data['flow_size'][idx] > 0 and demand_data['sn'][idx] != demand_data['dn'][idx]:
            arrival_times.append(demand_data['event_time'][idx])
        else:
            pass
    if len(arrival_times) == 0:
        raise Exception('Could not find first event establish request with size > 0.. This occurs because either demand_data given does not contain any events, or because all events have had to be dropped to try get below your specified target load. Try increasing the target load or increasing the granularity of load per demand (by e.g. decreasing demand sizes, increasing total number of demands, etc.) when you generate your demand data so that this function can more easily hit your desired load target.')

    time_first_flow_arrived = min(arrival_times)
    time_last_flow_arrived = max(arrival_times)
    
    return time_first_flow_arrived, time_last_flow_arrived



def duplicate_demands_in_demand_data_dict(demand_data, num_duplications=1, **kwargs):
    '''Duplicates set of demands by the specified number of times.'''
    demand_data = copy.deepcopy(demand_data)

    if 'use_multiprocessing' not in kwargs:
        kwargs['use_multiprocessing'] = False
    if 'num_processes' not in kwargs:
        # increase to decrease processing time, decrease to decrease risk of memory errors
        kwargs['num_processes'] = 10 # num processes to run in parallel if multiprocessing
    if 'maxtasksperchild' not in kwargs:
        kwargs['maxtasksperchild'] = 1 # num tasks per process
    
    if 'job_id' in demand_data:
        job_centric = True
    else:
        job_centric = False

    # ensure values of dict are lists
    for key, value in demand_data.items():
        demand_data[key] = list(value)

    init_num_demands = len(demand_data['event_time'])
    demands_to_add = ((2**num_duplications)*init_num_demands) - init_num_demands

    # progress bar
    if not kwargs['use_multiprocessing']:
        # TODO: Get progress bar working for multiprocessing
        pbar = tqdm(total=demands_to_add, 
                desc='Duplicating demands',
                    miniters=1, 
                    # mininterval=1,
                    # maxinterval=1, # 2
                    leave=False,
                    smoothing=1e-5) # 1
    # pbar = tqdm(total=demands_to_add, 
            # desc='Duplicating demands',
            # miniters=1)
                # # mininterval=1,
                # # maxinterval=1, # 2
                # # leave=False,
                # # smoothing=1e-5) # 1

    start = time.time()
    for dup in range(num_duplications):

        # get curr num demands
        num_demands = len(demand_data['event_time'])

        # get curr duration
        duration = max(demand_data['event_time']) - min(demand_data['event_time']) 

        # duplicate demands
        if kwargs['use_multiprocessing']:
            # duplicate demands in parallel
            print('Duplication {} of {}...'.format(dup+1, num_duplications))
            s = time.time()

            # init shared lists for appending duplicated demands to
            jobs = multiprocessing.Manager().list()
            job_ids = multiprocessing.Manager().list()
            # unique_ids = multiprocessing.Manager().list()
            flow_ids = multiprocessing.Manager().list()
            sns = multiprocessing.Manager().list()
            dns = multiprocessing.Manager().list()
            flow_sizes = multiprocessing.Manager().list()
            event_times = multiprocessing.Manager().list()
            establishes = multiprocessing.Manager().list()
            indexes = multiprocessing.Manager().list()

            # for idx in trange(num_demands): 
                # results = process_map(duplicate_demand,
                                            # [demand_data['job'][idx], 
                                            # demand_data['sn'][idx],
                                            # demand_data['dn'][idx],
                                            # demand_data['flow_size'][idx],
                                            # demand_data['event_time'][idx],
                                            # duration,
                                            # demand_data['establish'][idx],
                                            # demand_data['index'][idx],
                                            # num_demands,
                                            # idx, 
                                            # jobs, 
                                            # job_ids,
                                            # unique_ids,
                                            # flow_ids,
                                            # sns,
                                            # dns,
                                            # flow_sizes,
                                            # event_times,
                                            # establishes,
                                            # indexes,
                                            # job_centric])
            # duplicate demands in parallel
            pool = multiprocessing.Pool(kwargs['num_processes'], maxtasksperchild=kwargs['maxtasksperchild'])
            results = [pool.apply_async(duplicate_demand,
                                        args=(
                                        demand_data['job'][idx], 
                                        demand_data['sn'][idx],
                                        demand_data['dn'][idx],
                                        demand_data['flow_size'][idx],
                                        demand_data['event_time'][idx],
                                        duration,
                                        demand_data['establish'][idx],
                                        demand_data['index'][idx],
                                        num_demands,
                                        idx, 
                                        jobs, 
                                        job_ids,
                                        # unique_ids,
                                        flow_ids,
                                        sns,
                                        dns,
                                        flow_sizes,
                                        event_times,
                                        establishes,
                                        indexes,
                                        job_centric))
                                       # callback=lambda _: pbar.update(1))
                                        for idx in range(num_demands)]
            pool.close()
            pool.join()
            del pool

            # collect duplicated demands from multiprocessing and add to demand_data
            if job_centric:
                demand_data['job_id'].extend(list(job_ids))
                demand_data['job'].extend(list(jobs))
            demand_data['flow_id'].extend(list(flow_ids))
            demand_data['sn'].extend(list(sns))
            demand_data['dn'].extend(list(dns))
            demand_data['flow_size'].extend(list(flow_sizes))
            demand_data['event_time'].extend(list(event_times))
            demand_data['establish'].extend(list(establishes))
            demand_data['index'].extend(list(indexes))

            e = time.time()
            print('Duplication completed in {} s.'.format(e-s))

        else:
            # not multiprocessing -> duplicate demands sequentially
            # no need to init separate lists since can append directly
            if job_centric:
                jobs = demand_data['job']
                job_ids = demand_data['job_id']
            else:
                jobs = None
                job_ids = None
            # unique_ids = []
            flow_ids = demand_data['flow_id']
            sns = demand_data['sn']
            dns = demand_data['dn']
            flow_sizes = demand_data['flow_size']
            event_times = demand_data['event_time']
            establishes = demand_data['establish']
            indexes = demand_data['index']
            if job_centric:
                demand = demand_data['job'][idx]
            else:
                demand = None # don't need
            for idx in range(num_demands):
                duplicate_demand(demand, 
                                 demand_data['sn'][idx],
                                 demand_data['dn'][idx],
                                 demand_data['flow_size'][idx],
                                 demand_data['event_time'][idx],
                                 duration,
                                 demand_data['establish'][idx],
                                 demand_data['index'][idx],
                                 num_demands,
                                 idx, 
                                 jobs, 
                                 job_ids,
                                 # unique_ids,
                                 flow_ids,
                                 sns,
                                 dns,
                                 flow_sizes,
                                 event_times,
                                 establishes,
                                 indexes,
                                 job_centric)

                # # collect duplicated demands and add to demand_data
                # if job_centric:
                    # demand_data['job_id'].extend(list(job_ids))
                    # demand_data['job'].extend(list(jobs))
                # demand_data['flow_id'].extend(list(flow_ids))
                # demand_data['sn'].extend(list(sns))
                # demand_data['dn'].extend(list(dns))
                # demand_data['flow_size'].extend(list(flow_sizes))
                # demand_data['event_time'].extend(list(event_times))
                # demand_data['establish'].extend(list(establishes))
                # demand_data['index'].extend(list(indexes))

                pbar.update(1)

    # make sure demand data still ordered in order of event time
    index = np.argsort(demand_data['event_time'])
    for key in demand_data.keys():
        if len(demand_data[key]) == len(index):
            # only index keys which are events (i.e. for job-centric, these are job keys, not flow keys)
            demand_data[key] = [demand_data[key][i] for i in index]

    if not kwargs['use_multiprocessing']:
        pbar.close()
    end = time.time()
    print('Duplicated from {} to {} total demands ({} duplication(s)) in {} s.'.format(init_num_demands, len(demand_data['event_time']), num_duplications, end-start))

    return demand_data 




def duplicate_demand(job, 
                     sn,
                     dn,
                     flow_size,
                     event_time,
                     duration,
                     establish,
                     index,
                     num_demands, 
                     idx, 
                     jobs,
                     job_ids,
                     # unique_ids,
                     flow_ids,
                     sns,
                     dns,
                     flow_sizes,
                     event_times,
                     establishes,
                     indexes,
                     job_centric=True):

    if job_centric:
        # job id
        job_id = int(idx + num_demands)
        job.graph['job_id'] = 'job_{}'.format(job_id)

        # attrs inside job
        # job = copy.deepcopy(demand_data['job'])[idx]
        flow_stats = {flow: job.get_edge_data(flow[0], flow[1]) for flow in job.edges} 
        for flow in flow_stats:
            # grab attr_dict for flow
            attr_dict = flow_stats[flow]['attr_dict']

            # update ids
            attr_dict['job_id'] = 'job_{}'.format(idx+num_demands)
            attr_dict['unique_id'] = attr_dict['job_id'] + '_' + attr_dict['flow_id']

            # flow src, dst, & size
            # if data dependency, is a flow
            if attr_dict['dependency_type'] == 'data_dep':
                flow_ids.append(attr_dict['unique_id'])
                sns.append(attr_dict['sn'])
                dns.append(attr_dict['dn'])
                flow_sizes.append(attr_dict['flow_size'])

            # confirm updates
            edge = attr_dict['edge']
            job.add_edge(edge[0], edge[1], attr_dict=attr_dict)

        jobs.append(job)
        job_ids.append('job_{}'.format(job_id))


    else:
        flow_ids.append('flow_{}'.format(int(idx+num_demands)))
        flow_sizes.append(flow_size)
        sns.append(sn)
        dns.append(dn)

    event_times.append(duration + event_time)
    establishes.append(establish)
    indexes.append(index + num_demands)

    # time.sleep(0.1)







def gen_network_skewness_heat_maps(network, 
                                   num_skewed_nodes=[], 
                                   loads=[], 
                                   roof_skew_factor=2,
                                   labeled_grid_resolution=0.05,
                                   min_skewed_traffic_requested=0.0, 
                                   max_skewed_traffic_requested=1.0,
                                   print_data=False,
                                   path_to_save=None,
                                   **kwargs):
    '''

    Assumptions:
        - All endpoints in network have equal bandwidth capacity, and this
            capacity is 1/n of the overall network capacity, where n is the
            number of endpoints in the network.
        - Skewness is simply a skew factor, which is the fractional difference 
            between traffic_per_skewed_node and traffic_per_non_skewed_node
        - All nodes are either skewed (by the same amount) or not skewed (by the same amount)

    Args:
        network (object)
        num_skewed_nodes (list): List of number of nodes to skew. Forms y-axis
            on heat map.
        loads (list): List of overall loads to apply to network (length of loads
            list determines the number of heat maps generated).
        min_skewed_traffic_requested (float): Minimum proportion of overall traffic
            requested by skewed nodes. Must be between 0 and 1. Forms minimum of
            heat map x-axis (matrix rows).
        max_skewed_traffic_requested (float): Maximum proportion of overall traffic
            requested by skewed nodes. Must be between 0 and 1. Forms maximum of
            heat map x-axis (matrix columns).
        show_fig (bool): Whether or not to display colour map.

    Returns skewed_nodes_traffic_requested (y-axis coords), proportion_nodes_skewed
    (x-axis coords), heat_maps (2D matrix of skewness values for x-y coords), and
    list of colour map matplotlib figures.

    '''
    if len(num_skewed_nodes) == 0:
        num_skewed_nodes = np.arange(0, len(network.graph['endpoints'])+1)
    proportion_nodes_skewed = np.asarray([num_skewed/len(network.graph['endpoints']) for num_skewed in num_skewed_nodes])

    step_size = (max_skewed_traffic_requested-min_skewed_traffic_requested)/(len(num_skewed_nodes)-1)
    skewed_nodes_traffic_requested = np.arange(min_skewed_traffic_requested, max_skewed_traffic_requested+step_size, step_size)
    max_idx = len(skewed_nodes_traffic_requested) - 1

    # pcolormesh requires X and Y to be 1 larger dimension than Z or will cut out values
    skewed_nodes_traffic_requested = np.append(skewed_nodes_traffic_requested, (skewed_nodes_traffic_requested[1]-skewed_nodes_traffic_requested[0])+skewed_nodes_traffic_requested[-1])
    proportion_nodes_skewed = np.append(proportion_nodes_skewed, (proportion_nodes_skewed[1]-proportion_nodes_skewed[0])+proportion_nodes_skewed[-1])

    if len(loads) == 0:
        loads = np.arange(0.1, 1.0, 0.1)
        loads = [round(load, 2) for load in loads]

    max_traffic_per_ep = 1/len(network.graph['endpoints'])
    if print_data:
        print('max traffic per ep: {}'.format(max_traffic_per_ep))

    # max possible node skew is where have highest network load w/ lowest no. skewed nodes w/ highest traffic per skew
    # max_node_skew = max(x for x in skewed_nodes_traffic_requested if x != 0) * max(x for x in loads if x != 0) * min(x for x in num_skewed_nodes if x != 0)
    # min_node_skew = min(x for x in skewed_nodes_traffic_requested if x != 0) * min(x for x in loads if x != 0) * max(x for x in num_skewed_nodes if x != 0)
    # print(min_node_skew, max_node_skew)

    figs = []
    heat_maps = {load: None for load in loads}
    for load in loads:
        if print_data:
            print('\n\nload {}'.format(load))
        # rows (y) -> skewed nodes, columns (x) -> traffic requested
        heat_map = np.zeros((len(num_skewed_nodes), len(skewed_nodes_traffic_requested)))
        for node_idx, num_skewed in enumerate(num_skewed_nodes):
            for traffic_idx, traffic_all_skewed_nodes in enumerate(skewed_nodes_traffic_requested):
                traffic_per_skewed_node = (traffic_all_skewed_nodes / num_skewed) * load
                traffic_per_non_skewed_node = ((1-traffic_all_skewed_nodes) / (len(network.graph['endpoints'])-num_skewed)) * load
                if print_data:
                    print('num skewed nodes: {} | traffic all skewed nodes: {}'.format(num_skewed, traffic_all_skewed_nodes))
                    print('traffic per skewed node: {} | traffic per non skewed node: {}'.format(traffic_per_skewed_node, traffic_per_non_skewed_node))
                if traffic_per_skewed_node > max_traffic_per_ep:
                    if print_data:
                        print('skewed nodes excess! distribute excess amongst non-skewed nodes')
                    # distribute excess amongst non-skewed nodes
                    excess_per_node = traffic_per_skewed_node - max_traffic_per_ep
                    total_excess = excess_per_node * num_skewed
                    if print_data:
                        print('excess per node: {} | total excess: {}'.format(excess_per_node, total_excess))
                    traffic_per_non_skewed_node += (total_excess/(len(network.graph['endpoints'])-num_skewed))
                    traffic_per_skewed_node = max_traffic_per_ep
                    if print_data:
                        print('traffic per skewed node: {} | traffic per non skewed node: {}'.format(traffic_per_skewed_node, traffic_per_non_skewed_node))
                if traffic_per_non_skewed_node > max_traffic_per_ep:
                    if print_data:
                        print('non-skewed nodes excess! distribute excess amongst skewed nodes')
                    # distribute excess amongst skewed nodes:
                    excess_per_node = traffic_per_non_skewed_node - max_traffic_per_ep
                    total_excess = excess_per_node * (len(network.graph['endpoints'])-num_skewed)
                    if print_data:
                        print('excess per node: {} | total excess: {}'.format(excess_per_node, total_excess))
                    traffic_per_skewed_node += (total_excess/num_skewed)
                    traffic_per_non_skewed_node = max_traffic_per_ep
                    if print_data:
                        print('traffic per skewed node: {} | traffic per non skewed node: {}'.format(traffic_per_skewed_node, traffic_per_non_skewed_node))
                # skew = abs(traffic_per_skewed_node - traffic_per_non_skewed_node) / (max_traffic_per_ep)

                # roof = 2 # factor beyond which we say the traffic is 'very skewed'
                # skew = min(max([traffic_per_skewed_node, traffic_per_non_skewed_node]) / min([traffic_per_skewed_node, traffic_per_non_skewed_node]), roof)

                skew = max([traffic_per_skewed_node, traffic_per_non_skewed_node]) / min([traffic_per_skewed_node, traffic_per_non_skewed_node])
                # if np.isnan(skew) or np.isinf(skew):
                if np.isnan(skew):
                    skew = 1
                if print_data:
                    print('skew: {} (node idx {} traff idx {} load {})\n'.format(skew, node_idx, traffic_idx, load))
                heat_map[max_idx-node_idx, max_idx-traffic_idx] = skew

        # store heat map
        heat_maps[load] = heat_map
        if path_to_save is not None:
            tools.pickle_data(path_to_save+'heat_map_load_{}'.format(load), heat_map)

        # plot heat map
        kwargs['title'] = 'Load {}'.format(load)
        if path_to_save is not None:
            kwargs['path_to_save'] = path_to_save+'heat_map_load_{}.png'.format(load)
        figs.append(plot_dists.plot_heat_map(proportion_nodes_skewed, skewed_nodes_traffic_requested, heat_map, roof_skew_factor, **kwargs))

        if kwargs['plot_labeled_heat_map']:
            # plot annotated grid of skew values
            if path_to_save is not None:
                kwargs['path_to_save'] = path_to_save+'labeled_heat_map_load_{}.png'.format(load)
            figs.append(plot_dists.plot_labeled_heat_map(proportion_nodes_skewed, skewed_nodes_traffic_requested, heat_map, labeled_grid_resolution, **kwargs))


    return skewed_nodes_traffic_requested, proportion_nodes_skewed, heat_maps, figs





            



    
















        

        












# FUNCTIONS

def get_flow_centric_demand_data_ep_load_rate(demand_data, ep, eps, method='all_eps'):
    '''
    If method=='all_eps', duration is time_last_flow_arrived-time_first_flow_arrived
    across all endpoints. If method=='per_ep', duration is time_last_flow_arrived-time_first_flow_arrived
    for this specific ep.
    '''
    ep_info = group_demand_data_into_ep_info(demand_data, eps)
    total_info = sum(ep_info[ep]['flow_size'])
    # if method == 'per_ep':
        # time_first_flow_arrived = min(ep_info[ep]['event_time'])
        # time_last_flow_arrived = max(ep_info[ep]['event_time'])
    # if method == 'all_eps':
    time_first_flow_arrived = min(demand_data['event_time'])
    time_last_flow_arrived = max(demand_data['event_time'])
    duration = time_last_flow_arrived - time_first_flow_arrived
    if duration != 0:
        load_rate = total_info / duration
    else:
        load_rate = float('inf')
    
    return load_rate

def get_flow_centric_demand_data_overall_load_rate(demand_data):
    '''
    If method == 'mean_per_ep', will calculate the total network load as being the mean
    average load on each endpoint link (i.e. sum info requests for each link ->
    find load of each link -> find mean of ep link loads)

    If method == 'mean_all_eps', will calculate the total network load as being
    the average load over all endpoint links (i.e. sum info requests for all links
    -> find overall load of network)
    '''
    info_arrived = get_flow_centric_demand_data_total_info_arrived(demand_data)
    first_event_time, last_event_time = get_first_last_flow_arrival_times(demand_data)
    duration = last_event_time - first_event_time

    if duration != 0:
        load_rate = info_arrived/duration
    else:
        load_rate = float('inf')

    return load_rate



def group_demand_data_into_ep_info(demand_data, eps):
    nested_dict = lambda: defaultdict(nested_dict)
    ep_info = nested_dict()
    added_flow = {flow_id: False for flow_id in demand_data['flow_id']}
    for ep in eps:
        ep_info[ep]['flow_size'] = []
        # ep_info[ep]['event_time'] = []
        ep_info[ep]['demand_data_idx'] = []
        ep_info[ep]['flow_id'] = []
        # ep_info[ep]['establish'] = []
        # ep_info[ep]['index'] = []
        ep_info[ep]['sn'] = []
        ep_info[ep]['dn'] = []
    # group demand data by ep
    for idx in range(len(demand_data['flow_id'])):
        if not added_flow[demand_data['flow_id'][idx]]: 
            # not yet added this flow
            ep_info[demand_data['sn'][idx]]['flow_size'].append(demand_data['flow_size'][idx])
            ep_info[demand_data['dn'][idx]]['flow_size'].append(demand_data['flow_size'][idx])
            # ep_info[demand_data['sn'][idx]]['event_time'].append(demand_data['event_time'][idx])
            # ep_info[demand_data['dn'][idx]]['event_time'].append(demand_data['event_time'][idx])
            ep_info[demand_data['sn'][idx]]['demand_data_idx'].append(idx)
            ep_info[demand_data['dn'][idx]]['demand_data_idx'].append(idx)
            ep_info[demand_data['sn'][idx]]['flow_id'].append(demand_data['flow_id'][idx])
            ep_info[demand_data['dn'][idx]]['flow_id'].append(demand_data['flow_id'][idx])
            # ep_info[demand_data['sn'][idx]]['establish'].append(demand_data['establish'][idx])
            # ep_info[demand_data['dn'][idx]]['establish'].append(demand_data['establish'][idx])
            # ep_info[demand_data['sn'][idx]]['index'].append(demand_data['index'][idx])
            # ep_info[demand_data['dn'][idx]]['index'].append(demand_data['index'][idx])
            ep_info[demand_data['sn'][idx]]['sn'].append(demand_data['sn'][idx])
            ep_info[demand_data['sn'][idx]]['dn'].append(demand_data['dn'][idx])
            ep_info[demand_data['dn'][idx]]['sn'].append(demand_data['sn'][idx])
            ep_info[demand_data['dn'][idx]]['dn'].append(demand_data['dn'][idx])
        else:
            # already added this flow
            pass

    return ep_info


def get_flow_centric_demand_data_total_info_arrived(demand_data): 
    info_arrived = 0
    # print('flow size {} {}'.format(type(demand_data['flow_size']), type(demand_data['flow_size'][0])))
    # if 'job_id' in demand_data:
        # # jobcentric
        # flow_data = demand_data['flow_data']
    # else:
        # # flowcentric
        # flow_data = demand_data

    for flow_size in demand_data['flow_size']:
        if flow_size > 0:
            info_arrived += flow_size
        else:
            pass
    
    return info_arrived
            
def get_first_last_flow_arrival_times(demand_data):
    arrival_times = []
    for idx in range(len(demand_data['event_time'])):
        if demand_data['flow_size'][idx] > 0 and demand_data['sn'][idx] != demand_data['dn'][idx]:
            arrival_times.append(demand_data['event_time'][idx])
        else:
            pass
    if len(arrival_times) == 0:
        raise Exception('Could not find first event establish request with size > 0.. This occurs because either demand_data given does not contain any events, or because all events have had to be dropped to try get below your specified target load. Try increasing the target load or increasing the granularity of load per demand (by e.g. decreasing demand sizes, increasing total number of demands, etc.) when you generate your demand data so that this function can more easily hit your desired load target.')

    time_first_flow_arrived = min(arrival_times)
    time_last_flow_arrived = max(arrival_times)
    
    return time_first_flow_arrived, time_last_flow_arrived



def duplicate_demands_in_demand_data_dict(demand_data, num_duplications=1, **kwargs):
    '''Duplicates set of demands by the specified number of times.'''
    demand_data = copy.deepcopy(demand_data)

    if 'use_multiprocessing' not in kwargs:
        kwargs['use_multiprocessing'] = False
    if 'num_processes' not in kwargs:
        # increase to decrease processing time, decrease to decrease risk of memory errors
        kwargs['num_processes'] = 10 # num processes to run in parallel if multiprocessing
    if 'maxtasksperchild' not in kwargs:
        kwargs['maxtasksperchild'] = 1 # num tasks per process
    
    if 'job_id' in demand_data:
        job_centric = True
    else:
        job_centric = False

    # ensure values of dict are lists
    for key, value in demand_data.items():
        demand_data[key] = list(value)

    init_num_demands = len(demand_data['event_time'])
    demands_to_add = ((2**num_duplications)*init_num_demands) - init_num_demands

    # progress bar
    if not kwargs['use_multiprocessing']:
        # TODO: Get progress bar working for multiprocessing
        pbar = tqdm(total=demands_to_add, 
                desc='Duplicating demands',
                    miniters=1, 
                    # mininterval=1,
                    # maxinterval=1, # 2
                    leave=False,
                    smoothing=1e-5) # 1
    # pbar = tqdm(total=demands_to_add, 
            # desc='Duplicating demands',
            # miniters=1)
                # # mininterval=1,
                # # maxinterval=1, # 2
                # # leave=False,
                # # smoothing=1e-5) # 1

    start = time.time()
    for dup in range(num_duplications):

        # get curr num demands
        num_demands = len(demand_data['event_time'])

        # get curr duration
        duration = max(demand_data['event_time']) - min(demand_data['event_time']) 

        # duplicate demands
        if kwargs['use_multiprocessing']:
            # duplicate demands in parallel
            print('Duplication {} of {}...'.format(dup+1, num_duplications))
            s = time.time()

            # init shared lists for appending duplicated demands to
            jobs = multiprocessing.Manager().list()
            job_ids = multiprocessing.Manager().list()
            # unique_ids = multiprocessing.Manager().list()
            flow_ids = multiprocessing.Manager().list()
            sns = multiprocessing.Manager().list()
            dns = multiprocessing.Manager().list()
            flow_sizes = multiprocessing.Manager().list()
            event_times = multiprocessing.Manager().list()
            establishes = multiprocessing.Manager().list()
            indexes = multiprocessing.Manager().list()

            # for idx in trange(num_demands): 
                # results = process_map(duplicate_demand,
                                            # [demand_data['job'][idx], 
                                            # demand_data['sn'][idx],
                                            # demand_data['dn'][idx],
                                            # demand_data['flow_size'][idx],
                                            # demand_data['event_time'][idx],
                                            # duration,
                                            # demand_data['establish'][idx],
                                            # demand_data['index'][idx],
                                            # num_demands,
                                            # idx, 
                                            # jobs, 
                                            # job_ids,
                                            # unique_ids,
                                            # flow_ids,
                                            # sns,
                                            # dns,
                                            # flow_sizes,
                                            # event_times,
                                            # establishes,
                                            # indexes,
                                            # job_centric])
            # duplicate demands in parallel
            pool = multiprocessing.Pool(kwargs['num_processes'], maxtasksperchild=kwargs['maxtasksperchild'])
            results = [pool.apply_async(duplicate_demand,
                                        args=(
                                        demand_data['job'][idx], 
                                        demand_data['sn'][idx],
                                        demand_data['dn'][idx],
                                        demand_data['flow_size'][idx],
                                        demand_data['event_time'][idx],
                                        duration,
                                        demand_data['establish'][idx],
                                        demand_data['index'][idx],
                                        num_demands,
                                        idx, 
                                        jobs, 
                                        job_ids,
                                        # unique_ids,
                                        flow_ids,
                                        sns,
                                        dns,
                                        flow_sizes,
                                        event_times,
                                        establishes,
                                        indexes,
                                        job_centric))
                                       # callback=lambda _: pbar.update(1))
                                        for idx in range(num_demands)]
            pool.close()
            pool.join()
            del pool

            # collect duplicated demands from multiprocessing and add to demand_data
            if job_centric:
                demand_data['job_id'].extend(list(job_ids))
                demand_data['job'].extend(list(jobs))
            demand_data['flow_id'].extend(list(flow_ids))
            demand_data['sn'].extend(list(sns))
            demand_data['dn'].extend(list(dns))
            demand_data['flow_size'].extend(list(flow_sizes))
            demand_data['event_time'].extend(list(event_times))
            demand_data['establish'].extend(list(establishes))
            demand_data['index'].extend(list(indexes))

            e = time.time()
            print('Duplication completed in {} s.'.format(e-s))

        else:
            # not multiprocessing -> duplicate demands sequentially
            # no need to init separate lists since can append directly
            if job_centric:
                jobs = demand_data['job']
                job_ids = demand_data['job_id']
            else:
                jobs = None
                job_ids = None
            # unique_ids = []
            flow_ids = demand_data['flow_id']
            sns = demand_data['sn']
            dns = demand_data['dn']
            flow_sizes = demand_data['flow_size']
            event_times = demand_data['event_time']
            establishes = demand_data['establish']
            indexes = demand_data['index']
            if job_centric:
                demand = demand_data['job'][idx]
            else:
                demand = None # don't need
            for idx in range(num_demands):
                duplicate_demand(demand, 
                                 demand_data['sn'][idx],
                                 demand_data['dn'][idx],
                                 demand_data['flow_size'][idx],
                                 demand_data['event_time'][idx],
                                 duration,
                                 demand_data['establish'][idx],
                                 demand_data['index'][idx],
                                 num_demands,
                                 idx, 
                                 jobs, 
                                 job_ids,
                                 # unique_ids,
                                 flow_ids,
                                 sns,
                                 dns,
                                 flow_sizes,
                                 event_times,
                                 establishes,
                                 indexes,
                                 job_centric)

                # # collect duplicated demands and add to demand_data
                # if job_centric:
                    # demand_data['job_id'].extend(list(job_ids))
                    # demand_data['job'].extend(list(jobs))
                # demand_data['flow_id'].extend(list(flow_ids))
                # demand_data['sn'].extend(list(sns))
                # demand_data['dn'].extend(list(dns))
                # demand_data['flow_size'].extend(list(flow_sizes))
                # demand_data['event_time'].extend(list(event_times))
                # demand_data['establish'].extend(list(establishes))
                # demand_data['index'].extend(list(indexes))

                pbar.update(1)

    # make sure demand data still ordered in order of event time
    index = np.argsort(demand_data['event_time'])
    for key in demand_data.keys():
        if len(demand_data[key]) == len(index):
            # only index keys which are events (i.e. for job-centric, these are job keys, not flow keys)
            demand_data[key] = [demand_data[key][i] for i in index]

    if not kwargs['use_multiprocessing']:
        pbar.close()
    end = time.time()
    print('Duplicated from {} to {} total demands ({} duplication(s)) in {} s.'.format(init_num_demands, len(demand_data['event_time']), num_duplications, end-start))

    return demand_data 




def duplicate_demand(job, 
                     sn,
                     dn,
                     flow_size,
                     event_time,
                     duration,
                     establish,
                     index,
                     num_demands, 
                     idx, 
                     jobs,
                     job_ids,
                     # unique_ids,
                     flow_ids,
                     sns,
                     dns,
                     flow_sizes,
                     event_times,
                     establishes,
                     indexes,
                     job_centric=True):

    if job_centric:
        # job id
        job_id = int(idx + num_demands)
        job.graph['job_id'] = 'job_{}'.format(job_id)

        # attrs inside job
        # job = copy.deepcopy(demand_data['job'])[idx]
        flow_stats = {flow: job.get_edge_data(flow[0], flow[1]) for flow in job.edges} 
        for flow in flow_stats:
            # grab attr_dict for flow
            attr_dict = flow_stats[flow]['attr_dict']

            # update ids
            attr_dict['job_id'] = 'job_{}'.format(idx+num_demands)
            attr_dict['unique_id'] = attr_dict['job_id'] + '_' + attr_dict['flow_id']

            # flow src, dst, & size
            # if data dependency, is a flow
            if attr_dict['dependency_type'] == 'data_dep':
                flow_ids.append(attr_dict['unique_id'])
                sns.append(attr_dict['sn'])
                dns.append(attr_dict['dn'])
                flow_sizes.append(attr_dict['flow_size'])

            # confirm updates
            edge = attr_dict['edge']
            job.add_edge(edge[0], edge[1], attr_dict=attr_dict)

        jobs.append(job)
        job_ids.append('job_{}'.format(job_id))


    else:
        flow_ids.append('flow_{}'.format(int(idx+num_demands)))
        flow_sizes.append(flow_size)
        sns.append(sn)
        dns.append(dn)

    event_times.append(duration + event_time)
    establishes.append(establish)
    indexes.append(index + num_demands)

    # time.sleep(0.1)







def gen_network_skewness_heat_maps(network, 
                                   num_skewed_nodes=[], 
                                   loads=[], 
                                   roof_skew_factor=2,
                                   labeled_grid_resolution=0.05,
                                   min_skewed_traffic_requested=0.0, 
                                   max_skewed_traffic_requested=1.0,
                                   print_data=False,
                                   path_to_save=None,
                                   **kwargs):
    '''

    Assumptions:
        - All endpoints in network have equal bandwidth capacity, and this
            capacity is 1/n of the overall network capacity, where n is the
            number of endpoints in the network.
        - Skewness is simply a skew factor, which is the fractional difference 
            between traffic_per_skewed_node and traffic_per_non_skewed_node
        - All nodes are either skewed (by the same amount) or not skewed (by the same amount)

    Args:
        network (object)
        num_skewed_nodes (list): List of number of nodes to skew. Forms y-axis
            on heat map.
        loads (list): List of overall loads to apply to network (length of loads
            list determines the number of heat maps generated).
        min_skewed_traffic_requested (float): Minimum proportion of overall traffic
            requested by skewed nodes. Must be between 0 and 1. Forms minimum of
            heat map x-axis (matrix rows).
        max_skewed_traffic_requested (float): Maximum proportion of overall traffic
            requested by skewed nodes. Must be between 0 and 1. Forms maximum of
            heat map x-axis (matrix columns).
        show_fig (bool): Whether or not to display colour map.

    Returns skewed_nodes_traffic_requested (y-axis coords), proportion_nodes_skewed
    (x-axis coords), heat_maps (2D matrix of skewness values for x-y coords), and
    list of colour map matplotlib figures.

    '''
    if len(num_skewed_nodes) == 0:
        num_skewed_nodes = np.arange(0, len(network.graph['endpoints'])+1)
    proportion_nodes_skewed = np.asarray([num_skewed/len(network.graph['endpoints']) for num_skewed in num_skewed_nodes])

    step_size = (max_skewed_traffic_requested-min_skewed_traffic_requested)/(len(num_skewed_nodes)-1)
    skewed_nodes_traffic_requested = np.arange(min_skewed_traffic_requested, max_skewed_traffic_requested+step_size, step_size)
    max_idx = len(skewed_nodes_traffic_requested) - 1

    # pcolormesh requires X and Y to be 1 larger dimension than Z or will cut out values
    skewed_nodes_traffic_requested = np.append(skewed_nodes_traffic_requested, (skewed_nodes_traffic_requested[1]-skewed_nodes_traffic_requested[0])+skewed_nodes_traffic_requested[-1])
    proportion_nodes_skewed = np.append(proportion_nodes_skewed, (proportion_nodes_skewed[1]-proportion_nodes_skewed[0])+proportion_nodes_skewed[-1])

    if len(loads) == 0:
        loads = np.arange(0.1, 1.0, 0.1)
        loads = [round(load, 2) for load in loads]

    max_traffic_per_ep = 1/len(network.graph['endpoints'])
    if print_data:
        print('max traffic per ep: {}'.format(max_traffic_per_ep))

    # max possible node skew is where have highest network load w/ lowest no. skewed nodes w/ highest traffic per skew
    # max_node_skew = max(x for x in skewed_nodes_traffic_requested if x != 0) * max(x for x in loads if x != 0) * min(x for x in num_skewed_nodes if x != 0)
    # min_node_skew = min(x for x in skewed_nodes_traffic_requested if x != 0) * min(x for x in loads if x != 0) * max(x for x in num_skewed_nodes if x != 0)
    # print(min_node_skew, max_node_skew)

    figs = []
    heat_maps = {load: None for load in loads}
    for load in loads:
        if print_data:
            print('\n\nload {}'.format(load))
        # rows (y) -> skewed nodes, columns (x) -> traffic requested
        heat_map = np.zeros((len(num_skewed_nodes), len(skewed_nodes_traffic_requested)))
        for node_idx, num_skewed in enumerate(num_skewed_nodes):
            for traffic_idx, traffic_all_skewed_nodes in enumerate(skewed_nodes_traffic_requested):
                traffic_per_skewed_node = (traffic_all_skewed_nodes / num_skewed) * load
                traffic_per_non_skewed_node = ((1-traffic_all_skewed_nodes) / (len(network.graph['endpoints'])-num_skewed)) * load
                if print_data:
                    print('num skewed nodes: {} | traffic all skewed nodes: {}'.format(num_skewed, traffic_all_skewed_nodes))
                    print('traffic per skewed node: {} | traffic per non skewed node: {}'.format(traffic_per_skewed_node, traffic_per_non_skewed_node))
                if traffic_per_skewed_node > max_traffic_per_ep:
                    if print_data:
                        print('skewed nodes excess! distribute excess amongst non-skewed nodes')
                    # distribute excess amongst non-skewed nodes
                    excess_per_node = traffic_per_skewed_node - max_traffic_per_ep
                    total_excess = excess_per_node * num_skewed
                    if print_data:
                        print('excess per node: {} | total excess: {}'.format(excess_per_node, total_excess))
                    traffic_per_non_skewed_node += (total_excess/(len(network.graph['endpoints'])-num_skewed))
                    traffic_per_skewed_node = max_traffic_per_ep
                    if print_data:
                        print('traffic per skewed node: {} | traffic per non skewed node: {}'.format(traffic_per_skewed_node, traffic_per_non_skewed_node))
                if traffic_per_non_skewed_node > max_traffic_per_ep:
                    if print_data:
                        print('non-skewed nodes excess! distribute excess amongst skewed nodes')
                    # distribute excess amongst skewed nodes:
                    excess_per_node = traffic_per_non_skewed_node - max_traffic_per_ep
                    total_excess = excess_per_node * (len(network.graph['endpoints'])-num_skewed)
                    if print_data:
                        print('excess per node: {} | total excess: {}'.format(excess_per_node, total_excess))
                    traffic_per_skewed_node += (total_excess/num_skewed)
                    traffic_per_non_skewed_node = max_traffic_per_ep
                    if print_data:
                        print('traffic per skewed node: {} | traffic per non skewed node: {}'.format(traffic_per_skewed_node, traffic_per_non_skewed_node))
                # skew = abs(traffic_per_skewed_node - traffic_per_non_skewed_node) / (max_traffic_per_ep)

                # roof = 2 # factor beyond which we say the traffic is 'very skewed'
                # skew = min(max([traffic_per_skewed_node, traffic_per_non_skewed_node]) / min([traffic_per_skewed_node, traffic_per_non_skewed_node]), roof)

                skew = max([traffic_per_skewed_node, traffic_per_non_skewed_node]) / min([traffic_per_skewed_node, traffic_per_non_skewed_node])
                # if np.isnan(skew) or np.isinf(skew):
                if np.isnan(skew):
                    skew = 1
                if print_data:
                    print('skew: {} (node idx {} traff idx {} load {})\n'.format(skew, node_idx, traffic_idx, load))
                heat_map[max_idx-node_idx, max_idx-traffic_idx] = skew

        # store heat map
        heat_maps[load] = heat_map
        if path_to_save is not None:
            tools.pickle_data(path_to_save+'heat_map_load_{}'.format(load), heat_map)

        # plot heat map
        kwargs['title'] = 'Load {}'.format(load)
        if path_to_save is not None:
            kwargs['path_to_save'] = path_to_save+'heat_map_load_{}.png'.format(load)
        figs.append(plot_dists.plot_heat_map(proportion_nodes_skewed, skewed_nodes_traffic_requested, heat_map, roof_skew_factor, **kwargs))

        if kwargs['plot_labeled_heat_map']:
            # plot annotated grid of skew values
            if path_to_save is not None:
                kwargs['path_to_save'] = path_to_save+'labeled_heat_map_load_{}.png'.format(load)
            figs.append(plot_dists.plot_labeled_heat_map(proportion_nodes_skewed, skewed_nodes_traffic_requested, heat_map, labeled_grid_resolution, **kwargs))


    return skewed_nodes_traffic_requested, proportion_nodes_skewed, heat_maps, figs





            



    
























