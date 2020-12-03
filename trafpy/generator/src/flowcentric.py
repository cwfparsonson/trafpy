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
    # TODO: Not sure what below function was for but generates strange rand var dists, should use gen_rand_vars_from_discretised_dist() instead.
    # flow_sizes[:num_demands] = val_dists.gen_val_dist_data(val_dist=list(flow_size_dist.values()), 
                                                           # num_vals_to_gen=num_demands, 
                                                           # min_val=min(flow_size_dist.keys()), 
                                                           # max_val=max(flow_size_dist.keys()))
    flow_sizes[:num_demands] = val_dists.gen_rand_vars_from_discretised_dist(unique_vars=list(flow_size_dist.keys()),
                                                                             probabilities=list(flow_size_dist.values()),
                                                                             num_demands=num_demands)
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



def duplicate_demands_in_demand_data_dict(demand_data):
    num_demands = len(demand_data['flow_id'])
    final_event_time = max(demand_data['event_time'])
    copy_demand_data = copy.deepcopy(demand_data) 

    # ensure values of dict are lists
    for key, value in copy_demand_data.items():
        copy_demand_data[key] = list(value)

    for idx in range(len(demand_data['flow_id'])):
        copy_demand_data['flow_id'].append('flow_{}'.format(int(idx+num_demands)))
        copy_demand_data['sn'].append(demand_data['sn'][idx])
        copy_demand_data['dn'].append(demand_data['dn'][idx])
        copy_demand_data['flow_size'].append(demand_data['flow_size'][idx])
        copy_demand_data['event_time'].append(final_event_time + demand_data['event_time'][idx])
        copy_demand_data['establish'].append(demand_data['establish'][idx])
        copy_demand_data['index'].append(demand_data['index'][idx] + idx)

    return copy_demand_data




def adjust_demand_load(demand_data,
                       network_load_config,
                       num_demands,
                       eps,
                       node_dist,
                       flow_size_dist,
                       interarrival_time_dist,
                       duration_time_dist,
                       print_data=False):
    # adjust to get load fraction > target load fraction
    demand_data, new_interarrival_time_dist, new_num_demands = increase_demand_load_to_target(demand_data, 
                                                                             num_demands,
                                                                             interarrival_time_dist, 
                                                                             eps, 
                                                                             node_dist, 
                                                                             flow_size_dist, 
                                                                             network_load_config, 
                                                                             print_data=print_data)

    # adjust back down to get load fraction <= target load fraction
    demand_data, new_interarrival_time_dist = decrease_demand_load_to_target(demand_data, 
                                                                        new_num_demands,
                                                                        interarrival_time_dist=new_interarrival_time_dist,
                                                                        eps=eps,
                                                                        node_dist=node_dist,
                                                                        flow_size_dist=flow_size_dist,
                                                                        network_load_config=network_load_config,
                                                                        print_data=True)
    return demand_data, interarrival_time_dist




def increase_demand_load_to_target(demand_data, 
                                   num_demands, 
                                   interarrival_time_dist, 
                                   eps, 
                                   node_dist, 
                                   flow_size_dist, 
                                   network_load_config, 
                                   increment_factor=0.5,
                                   print_data=False):
    load_rate = get_flow_centric_demand_data_load_rate(demand_data)
    load_fraction = load_rate / network_load_config['network_rate_capacity']

    num_loops = 1
    # adjust to get load fraction >= target load fraction
    while load_fraction < network_load_config['target_load_fraction']:

        # # increase number of demands by 1% to try increase loads
        # num_demands = int(1.01 * num_demands)

        # decrease interarrival times to try increase load
        new_interarrival_time_dist = {}
        for rand_var, prob in interarrival_time_dist.items():
            new_rand_var = rand_var * increment_factor
            new_interarrival_time_dist[new_rand_var] = prob

        # update interarrival time dist
        interarrival_time_dist = new_interarrival_time_dist

        demand_data = create_flow_centric_demand_data(num_demands=num_demands,
                                                                  eps=eps,
                                                                  node_dist=node_dist,
                                                                  flow_size_dist=flow_size_dist,
                                                                  interarrival_time_dist=interarrival_time_dist,
                                                                  print_data=print_data)
        load_rate = get_flow_centric_demand_data_load_rate(demand_data)
        load_fraction = load_rate / network_load_config['network_rate_capacity']
        num_loops += 1
        if print_data:
            print('Reached load of {} (target load {}) after {} loops.'.format(load_fraction, network_load_config['target_load_fraction'], num_loops))
        if network_load_config['disable_timeouts']:
            # keep running loop to infinity
            if num_loops % 10 == 0:
                if print_data:
                    print('Warning: Have disabled timeouts. Ran {} loops to try to reach {} network load (reached {} load so far). Set network_load_config[\'disable_timeouts\']=True if desired. Disable this warning by setting print_data=False when calling create_demand_data.'.format(num_loops, network_load_config['target_load_fraction'], load_fraction))
        else:
            if num_loops > 15:
                raise Exception('Time out trying to reach requested network load fraction (reached {} but requested {}). Consider adjusting demand data parameters (e.g. increase flow size, decrease interarrival time, etc.), decreasing target_load_fraction, or decreasing network_rate_capacity. Alternatively, to disable timeouts, set network_load_config[\'disable_timeouts\'] = True.'.format(load_fraction, network_load_config['target_load_fraction']))

    return demand_data, interarrival_time_dist, num_demands



def decrease_demand_load_to_target(demand_data, 
                                   num_demands, 
                                   interarrival_time_dist, 
                                   eps, 
                                   node_dist, 
                                   flow_size_dist, 
                                   network_load_config, 
                                   increment_factor=1.001,
                                   print_data=False):
    load_rate = get_flow_centric_demand_data_load_rate(demand_data)
    load_fraction = load_rate / network_load_config['network_rate_capacity']
    target_load_fraction = network_load_config['target_load_fraction']
    network_rate_capacity = network_load_config['network_rate_capacity']
    
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
                new_rand_var = rand_var * increment_factor
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

            # demand_data = drop_random_flow_from_demand_data(demand_data)
            load_rate = get_flow_centric_demand_data_load_rate(demand_data)
            # change_in_num_demands -= 1

            num_loops += 1
            if network_load_config['disable_timeouts']:
                # keep running loop to infinity
                if num_loops % 10 == 0:
                    if print_data:
                        print('Warning: Have disabled timeouts. Ran {} loops to try to reach {} network load (reached {} load so far). Set network_load_config[\'disable_timeouts\']=True if desired. Disable this warning by setting print_data=False when calling create_demand_data.'.format(num_loops, network_load_config['target_load_fraction'], load_fraction))
            else:
                if num_loops > 15:
                    raise Exception('Time out trying to reach requested network load fraction (reached {} but requested {}). Consider adjusting demand data parameters (e.g. increase flow size, decrease interarrival time, etc.), decreasing target_load_fraction, or decreasing network_rate_capacity. Alternatively, to disable timeouts, set network_load_config[\'disable_timeouts\'] = True.'.format(load_fraction, network_load_config['target_load_fraction']))

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

def get_flow_centric_demand_data_load_rate(demand_data, bidirectional_links=True):
    '''
    If flow connections are bidirectional_links, 1 flow takes up 2 endpoint links (the
    source link and the destination link), therefore effecitvely takes up load rate
    2*flow_size*duration bandwidth. If not bidriectional, only takes up
    1*flow_size*duration since only occupies bandwidth for 1 of these links.
    '''
    info_arrived = get_flow_centric_demand_data_total_info_arrived(demand_data)
    first_event_time, last_event_time = get_first_last_flow_arrival_times(demand_data)

    # print('first event: {} | last event: {} | total time: {}'.format(first_event_time, last_event_time, last_event_time-first_event_time))
    
    if bidirectional_links:
        load_rate = 2*info_arrived/(last_event_time-first_event_time)
    else:
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



