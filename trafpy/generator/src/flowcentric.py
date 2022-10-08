from trafpy.generator.src import tools
from trafpy.generator.src.dists import val_dists, node_dists, plot_dists
from trafpy.utils import get_class_from_path

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
                 # flow_packer_cls='trafpy.generator.src.packers.flow_packer_v1.FlowPackerV1',
                 flow_packer_cls='trafpy.generator.src.packers.flow_packer_v2.FlowPackerV2',
                 flow_packer_kwargs=None,
                 print_data=False,
                 **kwargs):
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
        self.flow_packer_cls = flow_packer_cls
        if flow_packer_kwargs is None:
            self.flow_packer_kwargs = {}
        else:
            self.flow_packer_kwargs = flow_packer_kwargs

        self.num_nodes, self.num_pairs, self.node_to_index, self.index_to_node = tools.get_network_params(self.eps)

        if self.network_load_config['target_load_fraction'] is not None:
            if self.network_load_config['target_load_fraction'] > 0.95:
                raise Exception('Target load fraction {} is invalid. Must be <= 0.95.'.format(self.network_load_config['target_load_fraction']))

        if not self.check_dont_exceed_one_ep_load:
            print('WARNING: check_dont_exceed_one_ep_load is set to False. This may result in end point loads going above 1.0, which for some users might be detrimental to the systems they want to test.')

    def create_flow_centric_demand_data(self, return_packing_time=False, return_packing_jensen_shannon_distance=False,):
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
        if isinstance(self.flow_packer_cls, str):
            # load packer class from string path
            flow_packer_cls = get_class_from_path(self.flow_packer_cls)
        else:
            flow_packer_cls = self.flow_packer_cls
        packer = flow_packer_cls(self,
                                 self.eps,
                                 self.node_dist,
                                 flow_ids,
                                 flow_sizes,
                                 interarrival_times,
                                 network_load_config=self.network_load_config,
                                 auto_node_dist_correction=self.auto_node_dist_correction,
                                 check_dont_exceed_one_ep_load=self.check_dont_exceed_one_ep_load,
                                 **self.flow_packer_kwargs,
                                 )
        packer.reset()
        packed_flows = packer.pack_the_flows()
        self.packing_time = packer.packing_time
        self.packing_jensen_shannon_distance = packer.packing_jensen_shannon_distance

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
        if not return_packing_time and not return_packing_jensen_shannon_distance:
            returns = demand_data
        else:
            returns = [demand_data]
            if return_packing_time:
                returns.append(self.packing_time)
            if return_packing_jensen_shannon_distance:
                returns.append(self.packing_jensen_shannon_distance)
        return returns

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

            # flow src, dst, and size
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





            



    
























