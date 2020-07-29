from trafpy.src import jobcentric 

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
        assert c is not None, 'Specify graph connectivity factor c if job centric'
    else:
        job_centric = True

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
    #else:
    #    demand_data = create_flow_centric_demand_data(node_dist,
    #                                                  flow_size_dist)

    if path_to_save is not None:
        val_dists.save_data(path_to_save, demand_data)
    
    return demand_data




