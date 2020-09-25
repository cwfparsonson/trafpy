from trafpy.generator.src.dists import val_dists, node_dists
from trafpy.generator.src import tools

import numpy as np
import time
import copy
import random


def create_flow_centric_demand_data(num_demands,
                                    eps,
                                    node_dist, 
                                    flow_size_dist,
                                    interarrival_time_dist,
                                    duration_time_dist=None,
                                    print_data=False):
    if print_data:
        print('Generating {} flow demands...'.format(num_demands))
    started = time.time()

    # initialise
    f_ids = ['flow_'+str(i) for i in range(num_demands)]
    if duration_time_dist is not None:
        # duplicate
        f_ids2 = ['flow_'+str(i) for i in range(num_demands)]
        flow_ids = f_ids + f_ids2 # duplicate
        establish = np.concatenate((np.ones((int(len(flow_ids)))), 
                                    np.zeros((int(len(flow_ids)))))) 
    else:
        flow_ids = f_ids
        establish = np.concatenate((np.ones((int(len(flow_ids)))), 
                                    np.zeros((int(len(flow_ids)))))) 

    flow_sizes = np.zeros((int(len(flow_ids))))
    
    if duration_time_dist is not None:
        duplicate=True
    else:
        duplicate=False
    sn, dn = node_dists.gen_node_demands(eps=eps,
                                         node_dist=node_dist, 
                                         num_demands=num_demands, 
                                         duplicate=duplicate)

    # create demand flow_sizes
    flow_sizes[:num_demands] = val_dists.gen_val_dist_data(val_dist=list(flow_size_dist.values()), 
                                                           num_vals_to_gen=num_demands, 
                                                           min_val=min(flow_size_dist.keys()), 
                                                           max_val=max(flow_size_dist.keys()))
    if duration_time_dist is not None:
        flow_sizes[num_demands:] = flow_sizes[:num_demands]
    
    # create event time array
    interarrival_times = val_dists.gen_rand_vars_from_discretised_dist(unique_vars=list(interarrival_time_dist.keys()),
                                                                       probabilities=list(interarrival_time_dist.values()),
                                                                       num_demands=num_demands)
    if duration_time_dist is not None:
        interarrival_times = val_dists.gen_rand_vars_from_discretised_dist(unique_vars=list(interarrival_time_dist.keys()),
                                                                           probabilities=list(interarrival_time_dist.values()),
                                                                           num_demands=num_demands)
    else:
        duration_times = None
    event_times = tools.gen_event_times(interarrival_times, duration_times)
    index, event_times_sorted = np.argsort(event_times), np.sort(event_times)

    # compile data into demand data dict
    demand_data = {'flow_id': np.array(flow_ids)[index],
                   'sn': sn[index],
                   'dn': dn[index],
                   'flow_size': flow_sizes[index],
                   'event_time': event_times_sorted,
                   'establish': establish[index].astype(int),
                   'index': index}
   
    ended = time.time()
    if print_data:
        print('Generated {} flow demands in {} seconds.'.format(num_demands,ended-started))

    return demand_data


def set_flow_centric_demand_data_network_load(demand_data, interarrival_time_dist, eps, node_dist, flow_size_dist, network_rate_capacity, target_load_fraction, print_data=False):
    demand_data = copy.deepcopy(demand_data)
    num_demands = len(demand_data['flow_size'])
    
    assert target_load_fraction <= 1, \
        'Must have target load fraction <= 1 for compatability with network rate capacity.'
    
    target_load_rate = network_rate_capacity*target_load_fraction
    load_rate = get_flow_centric_demand_data_load_rate(demand_data)
    init_load_rate = copy.deepcopy(load_rate)
    init_load_fraction = init_load_rate / network_rate_capacity

    change_in_num_demands = 0
    if load_rate > target_load_rate:
        # randomly drop demands and increase interarrival time dist until get target load rate
        num_loops = 1
        while load_rate > target_load_rate:
            # adjust interarrival dist
            new_interarrival_time_dist = {}
            for rand_var, prob in interarrival_time_dist.items():
                new_rand_var = rand_var * 1.01
                new_interarrival_time_dist[new_rand_var] = prob

            # update interarrival time dist
            interarrival_time_dist = new_interarrival_time_dist

            # re-create
            demand_data = create_flow_centric_demand_data(num_demands=num_demands,
                                                                      eps=eps,
                                                                      node_dist=node_dist,
                                                                      flow_size_dist=flow_size_dist,
                                                                      interarrival_time_dist=interarrival_time_dist,
                                                                      print_data=print_data)

            demand_data = drop_random_flow_from_demand_data(demand_data)
            load_rate = get_flow_centric_demand_data_load_rate(demand_data)
            change_in_num_demands -= 1

    elif load_rate < target_load_rate:
        load_fraction = load_rate / network_rate_capacity
        raise Exception('Load is {}, but requested target load is {}. Either decrease target load or increase load in demand_data. To increase load in demand_data, increase number of demands generated or adjust e.g. event size, interarrival time, etc., then re-construct demand_data and pass back into this function.'.format(load_fraction, target_load_fraction))
    else:
        pass
    
    load_fraction = load_rate / network_rate_capacity
    
    if print_data:
        print('Network rate capacity: {} Gbps'.format(network_rate_capacity))
        print('Initial load rate: {} Gbps'.format(init_load_rate))
        print('Initial load fraction: {}'.format(init_load_fraction))
        print('Target load rate: {} Gbps'.format(target_load_rate))
        print('Target load fraction: {}'.format(target_load_fraction))
        print('Final load rate: {} Gbps'.format(load_rate))
        print('Final load fraction: {}'.format(load_fraction))
        print('Change in number of demands: {}'.format(change_in_num_demands))
        print('Final number of demands: {}'.format(len(demand_data['flow_id'])))
    
    return demand_data, new_interarrival_time_dist

def drop_random_flow_from_demand_data(demand_data):
    event_indices = [i for i in range(len(demand_data['event_time']))]
    flow_idx_to_drop = random.choice(event_indices)
    
    num_loops = 0
    while demand_data['flow_size'][flow_idx_to_drop] < 0 or demand_data['flow_size'][flow_idx_to_drop] == 0:
        flow_idx_to_drop += 1
        if flow_idx_to_drop > len(event_indices)-1:
            # start from beginning
            flow_idx_to_drop = 0
        num_loops += 1
        if num_loops > len(event_indices):
            raise Exception('Cannot find event in demand_data to drop.')
            
    for key in list(demand_data.keys()):        
        if type(demand_data[key]) == list:
            new_data = demand_data[key]
            del new_data[flow_idx_to_drop]
        else:
            # is numpy array
            new_data = demand_data[key]
            new_data = np.delete(new_data, flow_idx_to_drop)
        demand_data[key] = new_data
            
    return demand_data

def get_flow_centric_demand_data_load_rate(demand_data):
    info_arrived = get_flow_centric_demand_data_total_info_arrived(demand_data)
    first_event_time, last_event_time = get_first_last_flow_arrival_times(demand_data)

    # print('first event: {} | last event: {} | total time: {}'.format(first_event_time, last_event_time, last_event_time-first_event_time))
    
    load_rate = info_arrived/(last_event_time-first_event_time)
    
    return load_rate



def get_flow_centric_demand_data_total_info_arrived(demand_data): 
    info_arrived = 0
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



