from trafpy.generator.src.dists import val_dists, node_dists
from trafpy.generator.src import tools

import numpy as np
import time


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
