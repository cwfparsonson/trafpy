'''Module for building demand data dictionaries (flow- and job-centric).'''

from trafpy.generator.src import jobcentric, flowcentric 
from trafpy.generator.src import tools

import numpy as np
import time


def create_demand_data(eps,
                       node_dist,
                       flow_size_dist,
                       interarrival_time_dist,
                       num_demands=None,
                       network_load_config=None,
                       duration_time_dist=None,
                       num_ops_dist=None,
                       c=None,
                       use_multiprocessing=True,
                       print_data=False,
                       path_to_save=None):
    """Create demand data dictionary using given distributions.

    If num_ops_dist and c are left as None, return flow-centric demand data.
    Otherwise, return job-centric demand data.

    Args:
        eps (list): List of network endpoints.
        node_dist (numpy array): 2d matrix of source-destination probabilities
            of occurring
        flow_size_dist (dict): Probability distribution whose key-value pairs are 
            flow size value-probability pairs. 
        interarrival_time_dist (dict): Probability distribution whose key-value pairs are 
            interarrival time value-probability pairs. 
        num_demands (int): Number of demands to generate. If None, must specify
            network_load_config
        network_load_config (dict): Dict of form {'network_rate_capacity': <int/float>, 'target_load_fraction': <float>, 'disable_timeouts': <bool>},
            where network_rate_capacity is the maximum rate (in e.g. Gbps) at which
            information can be reliably transmitted over the communication network
            which the demand data will be inserted into, and where target_load_fraction
            is the fraction of the network rate capacity being requested by the demands
            (e.g. target_load_fraction=0.75 would generate demands which request
            a load that is 75% of the network rate capacity from the first to 
            the last demand arriving). disable_timeouts defines whether or not 
            to stop looping when trying to meet specified network load.
            If network_load_config is None, must specify num_demands
        duration_time_dist (dict): Probability distribution whose key-value pairs are 
            duration time value-probability pairs. If specified, half events
            returned will be 'take-down' events (establish==0). If left as None,
            all returned events will be 'connection establishment' events
            (establish==1).
        num_ops_dist (dict): Probability distribution whose key-value pairs are 
            number of operations (in a job) value-probability pairs. 
        c (int/float): Coefficient which determines job graph connectivity and
            therefore the number of edges in the job graph. Use this because, for
            large enough c and n (number of nodes), edge formation probability
            when using Erdos-Renyi random graph creation scales with the
            number of edges such that p=c*ln(n)/n, where graph diameter (and
            number of edges) scales with O(ln(n)). See
            https://www.cs.cmu.edu/~avrim/598/chap4only.pdf for more information.
        use_multiprocessing (bool): Whether or not to use multiprocessing when
            generating data. For generating large numbers of big job computation
            graphs, it is recommended to use multiprocessing.
        print_data (bool): whether or not to print extra information about the
            generated data (such as time to generate).
        path_to_save (str): Path to directory (with file name included) in which
            to save generated distribution. E.g. path_to_save='data/dists/my_dist'.

    Returns:
        dict: Generated demand data (either flow-centric or job-centric demand
        demand data depending on the args given to the function).

    """
    # check num demands and load config 
    if num_demands is None and network_load_config is None:
        raise Exception('Must specify either num_demands or network_load_config.')
    if num_demands is None:
        assert network_load_config is not None, \
            'Must define network_load_config if leaving num_demands as None.'
    if network_load_config is not None:
        assert num_demands is None, \
            'Cannot specify num_demands if network_load_config is not None.'

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
    

    if network_load_config is not None:
        # init a guess for number of demands needed to meet desired load
        num_demands = 1000


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
        if network_load_config is not None:
            load_rate = flowcentric.get_flow_centric_demand_data_load_rate(demand_data)
            load_fraction = load_rate / network_load_config['network_rate_capacity']
            num_loops = 0
            while load_fraction < network_load_config['target_load_fraction']:
                # increase number of demands to try increase loads
                num_demands *= 2
                demand_data = flowcentric.create_flow_centric_demand_data(num_demands=num_demands,
                                                                          eps=eps,
                                                                          node_dist=node_dist,
                                                                          flow_size_dist=flow_size_dist,
                                                                          interarrival_time_dist=interarrival_time_dist,
                                                                          duration_time_dist=duration_time_dist,
                                                                          print_data=print_data)
                load_rate = flowcentric.get_flow_centric_demand_data_load_rate(demand_data)
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
                        raise Exception('Time out trying to reach requested network load fraction (reached {} but requested {}). Consider adjusting demand data parameters (e.g. increase flow size, decrease interarrival time, etc.), decreasing target_load_fraction, or decreasing network_rate_capacity. Alternatively, to disable timeouts, set network_load_config[\'disable_timeouts\'] = True}.'.format(load_fraction, network_load_config['target_load_fraction']))

            # remove demands to get load fraction <= target load fraction
            demand_data = flowcentric.set_flow_centric_demand_data_network_load(demand_data, 
                                                                                network_rate_capacity=network_load_config['network_rate_capacity'],
                                                                                target_load_fraction=network_load_config['target_load_fraction'],
                                                                                print_data=print_data)


    if path_to_save is not None:
        tools.pickle_data(path_to_save, demand_data)
    
    return demand_data



def construct_demand_slots_dict(demand_data,
                                slot_size=0.1):
    '''Takes demand data (job-centric or flow-centric) and generates time-slot demand dictionaries.

    Often when simulating networks, it is useful to divide the arriving demands
    into time slots. This function uses the generated demand data event times
    and the specified slot size to divide when each demand arrives in the simulation
    into specific time slots.

    Returned dict keys are time slot boundary times and values are any demands
    which arrive in the time slot.

    Args:
        demand_data (dict): Generated demand data (either flow-centric or job-centric).
        slot_size (int/float): Time period of each time slot.

    Returns:
        dict: Dictionary containing the original demand data organised into time 
        slots.

    '''

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










