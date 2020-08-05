from trafpy.generator.src import jobcentric, flowcentric 
from trafpy.generator.src import tools

import numpy as np


def create_demand_data(num_demands,
                       eps,
                       node_dist,
                       flow_size_dist,
                       interarrival_time_dist,
                       duration_time_dist=None,
                       num_ops_dist=None,
                       c=None,
                       use_multiprocessing=True,
                       print_data=False,
                       path_to_save=None):
    """
    Creates a set of traffic demand data for a dynamic time series demand 
    scheme using a poisson process. N.B. First half of time array (and of
    source & destination nodes) are for connection establishment (i.e. have
    self.num_demands connection establishments), second half are for tearing
    down requests (i.e. have self.num_demands connection teardownds). 
    Therefore although have only self.num_demands requests/connections, 
    need to have self.num_demands*2 sources, destinations, and points in 
    time.

    Args:
    - self.num_demands (int, float): number of traffic requests
    - flow_sizes (array): array of self.num_demands flow_sizes for 
    self.num_demands demands
    - node_dist (array of floats): normalised demand distribution matrix 
    """
    # check if provided dists have tuple with fig as second element
    if type(node_dist) == tuple:
        node_dist = node_dist[0]
    assert type(node_dist) == np.ndarray, 'Invalid dist provided. Must be \
            dict with var-prob key-value pairs.'
    if type(flow_size_dist) == tuple:
        flow_size_dist = flow_size_dist[0]
    assert type(flow_size_dist) == dict, 'Invalid dist provided. Must be \
            dict with var-prob key-value pairs.'
    if type(interarrival_time_dist) == tuple:
        interarrival_time_dist = interarrival_time_dist[0]
    assert type(interarrival_time_dist) == dict, 'Invalid dist provided. Must be \
            dict with var-prob key-value pairs.'
    if duration_time_dist is None:
        pass
    else:
        if type(duration_time_dist) == tuple:
            duration_time_dist = duration_time_dist[0]
        assert type(duration_time_dist) == dict, 'Invalid dist provided. Must be \
                dict with var-prob key-value pairs.'
    if num_ops_dist is None:
        pass
    else:
        if type(num_ops_dist) == tuple:
            num_ops_dist = num_ops_dist[0]
        assert type(num_ops_dist) == dict, 'Invalid dist provided. Must be \
                dict with var-prob key-value pairs.'

    if num_ops_dist is None:
        job_centric = False
    else:
        job_centric = True
        assert c is not None, 'Specify graph connectivity factor c if job centric'

    array_sum = np.round(np.sum(list(flow_size_dist.values())),2)
    assert array_sum == 1, \
        'array must sum to 1, but is {}'.format(array_sum)
    
    matrix_sum = np.round(np.sum(node_dist),2)
    assert matrix_sum == 1, \
        'demand distribution matrix must sum to 1, but is {}'.format(matrix_sum)
    
    if job_centric:
        demand_data = jobcentric.create_job_centric_demand_data(num_demands=num_demands,
                                                                eps=eps,
                                                                node_dist=node_dist, 
                                                                flow_size_dist=flow_size_dist,
                                                                num_ops_dist=num_ops_dist,
                                                                c=c,
                                                                interarrival_time_dist=interarrival_time_dist,
                                                                duration_time_dist=duration_time_dist,
                                                                use_multiprocessing=use_multiprocessing,
                                                                print_data=print_data)
    else:
        demand_data = flowcentric.create_flow_centric_demand_data(num_demands=num_demands,
                                                                  eps=eps,
                                                                  node_dist=node_dist,
                                                                  flow_size_dist=flow_size_dist,
                                                                  interarrival_time_dist=interarrival_time_dist,
                                                                  duration_time_dist=duration_time_dist,
                                                                  print_data=print_data)

    if path_to_save is not None:
        tools.pickle_data(path_to_save, demand_data)
    
    return demand_data



def construct_demand_slots_dict(demand_data,
                                slot_size=0.1):
    if 'job_id' in demand_data:
        job_centric = True
    else:
        job_centric = False

    session_start_time = demand_data['event_time'][0]
    session_end_time = demand_data['event_time'][-1]
    total_session_time = session_end_time - session_start_time

    total_num_time_slots = int(total_session_time/slot_size)

    slot_times = np.arange(session_start_time,session_end_time,slot_size)
    if slot_times[-1] < session_end_time:
        # add extra time slot
        total_num_time_slots += 1
        slot_times = np.append(slot_times,slot_times[-1]+slot_size)

    # ensure slot times have specified number of decimal places
    dummy_slot_size = str(slot_size)
    num_decimals = dummy_slot_size[::-1].find('.')
    for slot_iter in range(len(slot_times)):
        slot_times[slot_iter] = np.round(slot_times[slot_iter], num_decimals)

    # init slot dict
    slot_dict = {slot_iter: {'lb_time': slot_times[slot_iter],
                             'ub_time': slot_times[slot_iter+1],
                             'new_event_dicts': []}
                    for slot_iter in range(total_num_time_slots)}

    event_iter = 0
    slot_iter = 0
    slot_time = slot_times[slot_iter+1]
    while slot_iter <= total_num_time_slots-1:
        try: 
            while demand_data['event_time'][event_iter] < slot_time:
                if job_centric:
                    # must process job to unpack each event
                    event_dict = jobcentric.gen_job_event_dict(demand_data,event_iter)
                else:
                    # flow is in itself an event w/ no need for job ids etc
                    event_dict = tools.gen_event_dict(demand_data,event_iter)
                slot_dict[slot_iter]['new_event_dicts'].append(event_dict)
                event_iter += 1
        except IndexError:
            break
        slot_iter += 1
        try:
            slot_time = slot_times[slot_iter+1]
        except IndexError:
            break
    
    return slot_dict










