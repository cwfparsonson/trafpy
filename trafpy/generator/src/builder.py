'''Module for building demand data dictionaries (flow- and job-centric).'''

from trafpy.generator.src import jobcentric, flowcentric
from trafpy.generator.src import tools

import numpy as np
import time
import json
import copy
import sys


def create_demand_data(eps,
                       node_dist,
                       flow_size_dist,
                       interarrival_time_dist,
                       network_load_config,
                       min_num_demands=6000,
                       max_num_demands=None,
                       jensen_shannon_distance_threshold=0.1,
                       check_dont_exceed_one_ep_load=True,
                       min_last_demand_arrival_time=None,
                       auto_node_dist_correction=False,
                       # flow_packer_cls='trafpy.generator.src.packers.flow_packer_v1.FlowPackerV1',
                       flow_packer_cls='trafpy.generator.src.packers.flow_packer_v2.FlowPackerV2',
                       flow_packer_kwargs=None,
                       num_ops_dist=None,
                       c=None,
                       use_multiprocessing=True,
                       print_data=False,
                       path_to_save=None,
                       return_packing_time=False,
                       return_packing_jensen_shannon_distance=False,
                       **kwargs):
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
        network_load_config (dict): Dict of form {'network_rate_capacity': <int/float>, 'target_load_fraction': <float>, 'disable_timeouts': <bool>, 'return_new_interarrival_time_dist': <bool>},
            where network_rate_capacity is the maximum rate (in e.g. Gbps) at which
            information can be reliably transmitted over the communication network
            which the demand data will be inserted into, and where target_load_fraction
            is the fraction of the network rate capacity being requested by the demands
            (e.g. target_load_fraction=0.75 would generate demands which request
            a load that is 75% of the network rate capacity from the first to 
            the last demand arriving). disable_timeouts defines whether or not 
            to stop looping when trying to meet specified network load. return_new_interarrival_time_dist
            defines whether or not to return the new interarrival time dist which
            was adjusted to meet the network node requested.
            If network_load_config is None, must specify num_demands
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
        min_num_demands (int): Minimum number of demands to generate (will increase
            beyond this if need to meet min_last_demand_arrival_time and/or to
            meet specified jensen_shannon_distance_threshold).
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
        min_last_demand_arrival_time (int, float): Minimum last time of arrival
            for final demand (helps user specify a minimum simulation time). Will
            keep doubling number of demands until get >= min_last_demand_arrival_time.
        auto_node_dist_correction (bool): If True, if a node dist and overall load
            is specified which is not valid since would lead to at least one endpoint
            link's load exceeding 1.0, TrafPy will automatically adjust the node 
            dist by distributing the excess load uniformally across all other valid
            end points. As such, as the network load tends to 1.0, the node dist
            will approach uniform with 1.0 load being requested on all end point links.
        print_data (bool): whether or not to print extra information about the
            generated data (such as time to generate).
        path_to_save (str): Path to directory (with file name included) in which
            to save generated distribution. E.g. path_to_save='data/dists/my_dist'.

    Returns:
        dict: Generated demand data (either flow-centric or job-centric demand
            demand data depending on the args given to the function).

        N.B. If network_load_config is not None, will return tuple of
        (demand_data, interarrival_time_dist) to give updated interarrival time dist
        needed to meet network_load_config requested.
            

    """

    if num_ops_dist is None:
        # flowcentric
        generator = flowcentric.FlowGenerator(eps,
                                              node_dist=node_dist,
                                              flow_size_dist=flow_size_dist,
                                              interarrival_time_dist=interarrival_time_dist,
                                              network_load_config=network_load_config,
                                              min_num_demands=min_num_demands,
                                              max_num_demands=max_num_demands,
                                              jensen_shannon_distance_threshold=jensen_shannon_distance_threshold,
                                              min_last_demand_arrival_time=min_last_demand_arrival_time,
                                              auto_node_dist_correction=auto_node_dist_correction,
                                              check_dont_exceed_one_ep_load=check_dont_exceed_one_ep_load,
                                              flow_packer_cls=flow_packer_cls,
                                              flow_packer_kwargs=flow_packer_kwargs,
                                              print_data=print_data,
                                              **kwargs) 
        return generator.create_flow_centric_demand_data(
                return_packing_time=return_packing_time, 
                return_packing_jensen_shannon_distance=return_packing_jensen_shannon_distance,
                )

    else:
        # jobcentric
        generator = jobcentric.JobGenerator(eps,
                                              node_dist=node_dist,
                                              flow_size_dist=flow_size_dist,
                                              interarrival_time_dist=interarrival_time_dist,
                                              num_ops_dist=num_ops_dist,
                                              c=c,
                                              network_load_config=network_load_config,
                                              min_num_demands=min_num_demands,
                                              max_num_demands=max_num_demands,
                                              jensen_shannon_distance_threshold=jensen_shannon_distance_threshold,
                                              min_last_demand_arrival_time=min_last_demand_arrival_time,
                                              use_multiprocessing=use_multiprocessing,
                                              auto_node_dist_correction=auto_node_dist_correction,
                                              flow_packer_cls=flow_packer_cls,
                                              flow_packer_kwargs=flow_packer_kwargs,
                                              check_dont_exceed_one_ep_load=check_dont_exceed_one_ep_load,
                                              print_data=print_data,
                                              **kwargs) 
        return generator.create_job_centric_demand_data(
                return_packing_time=return_packing_time, 
                return_packing_jensen_shannon_distance=return_packing_jensen_shannon_distance,
                )







    

    # TEMPORARILY COMMENT
    '''
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
    elif type(node_dist) == str:
        # loaded from json, convert
        node_dist = json.loads(node_dist)
        node_dist = np.array(node_dist)
    assert type(node_dist) == np.ndarray, 'Invalid dist provided. Must be \
            dict with var-prob key-value pairs.'
    if type(flow_size_dist) == tuple:
        flow_size_dist = flow_size_dist[0]
    elif type(flow_size_dist) == str:
        # loaded from json, convert
        flow_size_dist = json.loads(flow_size_dist)
        # convert str keys to float
        new_dict = {float(k): v for k, v in iter(flow_size_dist.items())}
        flow_size_dist = new_dict
    assert type(flow_size_dist) == dict, 'Invalid dist provided. Must be \
            dict with var-prob key-value pairs.'
    if type(interarrival_time_dist) == tuple:
        interarrival_time_dist = interarrival_time_dist[0]
    if type(interarrival_time_dist) == str:
        # loaded from json, convert
        interarrival_time_dist = json.loads(interarrival_time_dist)
        # convert str keys to float
        new_dict = {float(k): v for k, v in iter(interarrival_time_dist.items())}
        interarrival_time_dist = new_dict
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
        if 'disable_timeouts' not in network_load_config.keys():
            # assume not disabling timeouts
            network_load_config['disable_timeouts'] = False
        if 'return_new_interarrival_time_dist' not in network_load_config.keys():
            # assume not returning new interarrival_time_dist
            network_load_config['return_new_interarrival_time_dist'] = False

    if network_load_config is not None:
        # init a guess for number of demands needed to meet desired load
        num_pairs = int(((len(eps) ** 2)/2) - len(eps))
        num_demands = int(num_pairs * num_demands_factor)


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
            print('Reconfiguring interarrival times and number of demands to match requested network load config...')
            demand_data, new_interarrival_time_dist = flowcentric.adjust_demand_load(demand_data=demand_data,
                                                                                     network_load_config=network_load_config,
                                                                                     num_demands=num_demands,
                                                                                     eps=eps,
                                                                                     node_dist=node_dist,
                                                                                     flow_size_dist=flow_size_dist,
                                                                                     interarrival_time_dist=interarrival_time_dist,
                                                                                     duration_time_dist=duration_time_dist,
                                                                                     print_data=print_data)

        if min_last_demand_arrival_time is not None:
            print('Ensuring last event arrives at time >= min_last_demand_arrival_time...')
            while max(demand_data['event_time']) < min_last_demand_arrival_time:
                # load_rate = flowcentric.get_flow_centric_demand_data_load_rate(demand_data, method='mean_per_ep', eps=eps)
                load_rate = flowcentric.get_flow_centric_demand_data_overall_load_rate(demand_data, bidirectional_links=True)
                print('\nFinal event time: {} | Target final event time: {} | load rate: {}'.format(max(demand_data['event_time']), min_last_demand_arrival_time, load_rate))
                total_info = sum(demand_data['flow_size'])
                time_last_flow_arrived = max(demand_data['event_time'])
                rate = total_info / time_last_flow_arrived
                print('total info: {} | last flow: {} | rate: {}'.format(total_info, time_last_flow_arrived, rate))
                demand_data = flowcentric.duplicate_demands_in_demand_data_dict(demand_data, method='all_eps', eps=eps)

            





    if path_to_save is not None:
        tools.pickle_data(path_to_save, demand_data)

    if network_load_config is not None:
        if network_load_config['return_new_interarrival_time_dist']:
            return demand_data, new_interarrival_time_dist
        else:
            return demand_data

    else:
        return demand_data
    '''



def construct_demand_slots_dict(demand_data,
                                slot_size=0.1,
                                include_empty_slots=True,
                                print_info=False):
    '''Takes demand data (job-centric or flow-centric) and generates time-slot demand dictionaries.

    Often when simulating networks, it is useful to divide the arriving demands
    into time slots. This function uses the generated demand data event times
    and the specified slot size to divide when each demand arrives in the simulation
    into specific time slots.

    Returned dict keys are time slot boundary times and values are any demands
    which arrive in the time slot.

    Args:
        demand_data (dict): Generated demand data (either flow-centric or job-centric).
        slot_size (float): Time period of each time slot. MUST BE FLOAT!!
        include_empty_slots (bool): Whether or not to include empty (i.e. no flows arriving)
            slots in slots_dict values. If True, will have keys for all slots of simulation,
            but will have larger memory usage, making the slots_dict less scalable.

    Returns:
        dict: Dictionary containing the original demand data organised into time 
        slots.

    '''
    start = time.time()

    slot_size = float(slot_size)
    if type(slot_size) is not float:
        raise Exception('slot_size must be float (e.g. 1.0), but is {}'.format(slot_size))

    if 'job_id' in demand_data:
        job_centric = True
    else:
        job_centric = False

    session_start_time = demand_data['event_time'][0]
    session_end_time = demand_data['event_time'][-1]
    total_session_time = session_end_time - session_start_time

    total_num_time_slots = int(total_session_time/slot_size)
    if total_num_time_slots != 0:
        slot_times = np.arange(session_start_time,session_end_time,slot_size)
        if slot_times[-1] < session_end_time:
            # add extra time slot
            total_num_time_slots += 1
            slot_times = np.append(slot_times,slot_times[-1]+slot_size)
    else:
        # all flows arrived immediately
        slot_times = [0, slot_size]
        total_num_time_slots = 1

    # ensure slot times have specified number of decimal places
    dummy_slot_size = str(slot_size)
    num_decimals = dummy_slot_size[::-1].find('.')
    if num_decimals == -1:
        raise Exception('Given slot_size {} has invalid num_decimals of {}. Make sure slot_size is given as a float e.g. use slot_size=1.0 rather than slot_size=1'.format(slot_size, num_decimals))
    # for slot_iter in range(len(slot_times)):
    for slot_iter in range(len(slot_times)):
        slot_times[slot_iter] = np.round(slot_times[slot_iter], num_decimals)

    # init slot dict
    slot_dict = {slot_iter: {'lb_time': slot_times[slot_iter],
                             'ub_time': slot_times[slot_iter]+slot_size,
                             'new_event_dicts': []}
                         for slot_iter in range(len(slot_times))}

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

    original_num_keys = len(slot_dict.keys())
    # remove any slots which do not contain any new flows
    new_slot_dict = {}
    num_empty_slots = 0
    num_demands = 0
    for slot in slot_dict.keys():
        num_demands_arrived = len(slot_dict[slot]['new_event_dicts'])
        num_demands += num_demands_arrived
        if num_demands_arrived == 0:
            # no new demands arrived, do not add to new slot dict
            num_empty_slots += 1
        else:
            # new demands arrived this slot, add to new slot dict
            if not include_empty_slots:
                new_slot_dict[slot] = slot_dict[slot]
            else:
                # just counting
                pass
    if not include_empty_slots:
        slot_dict = copy.deepcopy(new_slot_dict)
    end = time.time()

    if print_info:
        num_slots = len(slot_dict.keys())
        num_redundant_slots = num_empty_slots
        frac_redundant_slots = round(num_redundant_slots/num_slots, 3)
        avrg_num_demands_per_slot = round(num_demands / num_slots, 3)
        print('Generated slot dict in {} s with slot size {} and total session time {} for {} demands.'.format(round(end-start, 4), slot_size, total_session_time, num_demands))
        # print('Approx memory size of slot dict: {} Bytes (N.B. This is very unreliable and depends on save format)'.format(sys.getsizeof(json.dumps(slot_dict))))
        print('Number of slots making up total session time: {}'.format(num_slots))
        print('Number of these slots in which no new demands arrived: {}'.format(num_empty_slots))
        print('Fraction of the {} total time slots from simulation start to finish in which no new demands arrive: {}'.format(num_slots, frac_redundant_slots))
        print('Average number of demands arriving per time slot: {}'.format(avrg_num_demands_per_slot))
        if not include_empty_slots:
            print('Number of keys in updated slot dict (after removing empty slots where no new demands arrived): {}'.format(num_slots))
        if avrg_num_demands_per_slot < 1:
            print('Notice: In simulation, the scheduler makes a decision at every time slot. Therefore the more time slots there are, the more processing overhead there is, and therefore the longer the simulation will take. If many of your slot sizes are redundant (i.e. no new flow information is arriving), it is advisable to increase the slot size -> decrease the number of slots -> decrease the number steps in the simulation -> decrease the simulation time. Conversely, if you have a high number of demands arriving per time slot, your scheduler will have a lower resolution to process the flows which can lead to poorer performance and more flows being perhaps unnecesserily dropped from your network. As a rule of thumb, having an average number of flows arriving per time slot of less than 1 is not needed.')


    # init general slot dict params which are useful for simulations
    keys = list(slot_dict.keys())
    slot_dict['slot_keys'] = keys
    slot_dict['slot_size'] = slot_size
    slot_dict['time_first_demand_arrived'] = session_start_time
    slot_dict['time_last_demand_arrived'] = session_end_time
    slot_dict['job_centric'] = job_centric
    slot_dict['num_control_deps'], slot_dict['num_data_deps'], slot_dict['num_flows'] = get_num_deps(demand_data, job_centric)
    if not job_centric:
        slot_dict['num_demands'] = len(demand_data['flow_id'])
    else:
        slot_dict['num_demands'] = len(demand_data['job_id'])

    
    return slot_dict


def get_num_deps(demand_data, job_centric):
    num_control_deps,num_data_deps,num_flows = 0, 0, 0

    if job_centric:
        # calc deps
        for job in demand_data['job']:
            num_control_deps += job.graph['num_control_deps']
            num_data_deps += job.graph['num_data_deps']
            # for op in job.nodes:
                # flows = job.out_edges(op)
                # for flow in flows:
                    # flow_stats = job.get_edge_data(flow[0],flow[1])
                    # src = job.nodes[flow[0]]['attr_dict']['machine']
                    # dst = job.nodes[flow[1]]['attr_dict']['machine']
                    # if flow_stats['attr_dict']['dependency_type'] == 'data_dep':
                        # num_data_deps+=1
                        # if src != dst:
                            # num_flows+=1
                    # else:
                        # num_control_deps+=1
        num_flows = num_data_deps

    else:
        # 1 demand == 1 flow, therefore no dependencies & each demand == flow
        num_flows = len(demand_data['flow_id'])
    
    return num_control_deps, num_data_deps, num_flows








