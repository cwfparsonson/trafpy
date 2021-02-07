from trafpy.generator.src.dists import val_dists, node_dists
from trafpy.generator.src import tools

import numpy as np
import time
import copy
import random
from collections import defaultdict # use for initialising arbitrary length nested dict


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



def duplicate_demands_in_demand_data_dict(demand_data, method='all_eps', **kwargs):
    '''
    If method == 'all_eps', will duplicate all demands by adding final event time
    over all endpoints to each event time

    if method == 'per_ep', will duplicate all demands by adding final even time
    for each endpoint's final event time

    '''
    copy_demand_data = copy.deepcopy(demand_data) 

    # ensure values of dict are lists
    for key, value in copy_demand_data.items():
        copy_demand_data[key] = list(value)

    if method == 'all_eps':
        # final_event_time = max(demand_data['event_time'])
        num_demands = len(demand_data['flow_id'])
        final_event_time = max(demand_data['event_time'])
        first_event_time = min(demand_data['event_time'])
        duration = final_event_time - first_event_time
        for idx in range(len(demand_data['flow_id'])):
            copy_demand_data['flow_id'].append('flow_{}'.format(int(idx+num_demands)))
            copy_demand_data['sn'].append(demand_data['sn'][idx])
            copy_demand_data['dn'].append(demand_data['dn'][idx])
            copy_demand_data['flow_size'].append(demand_data['flow_size'][idx])
            # copy_demand_data['event_time'].append(final_event_time + demand_data['event_time'][idx])
            copy_demand_data['event_time'].append(duration + demand_data['event_time'][idx])
            copy_demand_data['establish'].append(demand_data['establish'][idx])
            copy_demand_data['index'].append(demand_data['index'][idx] + idx)

    elif method == 'per_ep':
        original_ep_info = group_demand_data_into_ep_info(copy_demand_data, eps=kwargs['eps'])
        # idx_iterator = iter(range(num_demands))
        duplicated_flows = {flow_id: False for flow_id in copy_demand_data['flow_id']}
        print('Flows before duplication: {}'.format(len(copy_demand_data['flow_id'])))
        idx_iterator = iter(range(len(copy_demand_data['flow_id'])))
        for ep in kwargs['eps']:
            # ep_info = group_demand_data_into_ep_info(copy_demand_data, eps=kwargs['eps'])
            num_demands = len(original_ep_info[ep]['flow_id'])

            first_event_time = min(original_ep_info[ep]['event_time'])
            final_event_time = max(original_ep_info[ep]['event_time'])
            duration = final_event_time - first_event_time

            # DEBUG 
            total_info = sum(original_ep_info[ep]['flow_size'])
            num_flows = len(original_ep_info[ep]['flow_id'])
            load = total_info / (final_event_time - first_event_time)
            print('Init {} duration: {} total info: {} | load: {} | flows: {}'.format(ep, duration, total_info, load, num_flows))

            for ep_flow_idx in range(len(original_ep_info[ep]['flow_id'])):
                flow_id = original_ep_info[ep]['flow_id'][ep_flow_idx]
                if not duplicated_flows[flow_id]:
                    duplicated_flows[flow_id] = True
                    # not yet duplicated this flow
                    # i = find_index_of_int_in_str(flow_id)
                    # idx = int(flow_id[i:])
                    idx = next(idx_iterator)

                    # copy_demand_data['flow_id'].append('flow_{}'.format(int(idx+num_demands)))
                    copy_demand_data['flow_id'].append('flow_{}'.format(int(idx+len(demand_data['flow_id']))))
                    copy_demand_data['sn'].append(original_ep_info[ep]['sn'][ep_flow_idx])
                    copy_demand_data['dn'].append(original_ep_info[ep]['dn'][ep_flow_idx])
                    copy_demand_data['flow_size'].append(original_ep_info[ep]['flow_size'][ep_flow_idx])
                    copy_demand_data['event_time'].append(duration + original_ep_info[ep]['event_time'][ep_flow_idx])
                    copy_demand_data['establish'].append(original_ep_info[ep]['establish'][ep_flow_idx])
                    copy_demand_data['index'].append(original_ep_info[ep]['index'][ep_flow_idx] + idx)
                else:
                    # already duplicated this flow in copy_demand_data
                    pass
            # DEBUG
            _ep_info = group_demand_data_into_ep_info(copy_demand_data, eps=kwargs['eps'])
            final_event_time = max(_ep_info[ep]['event_time'])
            first_event_time = min(_ep_info[ep]['event_time'])
            total_info = sum(_ep_info[ep]['flow_size'])
            load = total_info / (final_event_time - first_event_time)
            num_flows = len(_ep_info[ep]['flow_id'])
            print('Adjusted {} duration: {} total info: {} | load: {} | flows: {}'.format(ep, duration, total_info, load, num_flows))
        print('Flows after duplication: {}'.format(len(copy_demand_data['flow_id'])))

    # ensure values of dict are lists
    for key, value in copy_demand_data.items():
        copy_demand_data[key] = list(value)



    return copy_demand_data




def find_index_of_int_in_str(string):
    idx = 0
    for char in string:
        try:
            int(char)
            return idx
        except ValueError:
            # char is not an int
            idx += 1
    raise Exception('Could not find an integer in the string {}'.format(string))


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

    # adjust further to ensure no endpoint link has load fraction >= endpoint link margin of e.g. 0.95 while ensuring still meet target load fraction
    demand_data = adjust_demand_load_to_ep_link_margin(demand_data, 
                                                                        new_num_demands,
                                                                        interarrival_time_dist=new_interarrival_time_dist,
                                                                        eps=eps,
                                                                        node_dist=node_dist,
                                                                        flow_size_dist=flow_size_dist,
                                                                        network_load_config=network_load_config,
                                                                        print_data=True)

    # organise data in demand_data in order of events arriving
    index, event_times_sorted = np.argsort(demand_data['event_time']), np.sort(demand_data['event_time'])
    demand_data = {'flow_id': np.array(demand_data['flow_id'])[index],
                   'sn': demand_data['sn'][index],
                   'dn': demand_data['dn'][index],
                   'flow_size': demand_data['flow_size'][index],
                   'event_time': event_times_sorted,
                   'establish': demand_data['establish'][index].astype(int),
                   'index': index}


    return demand_data, interarrival_time_dist


def group_demand_data_into_ep_info(demand_data, eps):
    nested_dict = lambda: defaultdict(nested_dict)
    ep_info = nested_dict()
    added_flow = {flow_id: False for flow_id in demand_data['flow_id']}
    for ep in eps:
        ep_info[ep]['flow_size'] = []
        ep_info[ep]['event_time'] = []
        ep_info[ep]['demand_data_idx'] = []
        ep_info[ep]['flow_id'] = []
        ep_info[ep]['establish'] = []
        ep_info[ep]['index'] = []
        ep_info[ep]['sn'] = []
        ep_info[ep]['dn'] = []
    # group demand data by ep
    for idx in range(len(demand_data['flow_id'])):
        if not added_flow[demand_data['flow_id'][idx]]: 
            # not yet added this flow
            ep_info[demand_data['sn'][idx]]['flow_size'].append(demand_data['flow_size'][idx])
            ep_info[demand_data['dn'][idx]]['flow_size'].append(demand_data['flow_size'][idx])
            ep_info[demand_data['sn'][idx]]['event_time'].append(demand_data['event_time'][idx])
            ep_info[demand_data['dn'][idx]]['event_time'].append(demand_data['event_time'][idx])
            ep_info[demand_data['sn'][idx]]['demand_data_idx'].append(idx)
            ep_info[demand_data['dn'][idx]]['demand_data_idx'].append(idx)
            ep_info[demand_data['sn'][idx]]['flow_id'].append(demand_data['flow_id'][idx])
            ep_info[demand_data['dn'][idx]]['flow_id'].append(demand_data['flow_id'][idx])
            ep_info[demand_data['sn'][idx]]['establish'].append(demand_data['establish'][idx])
            ep_info[demand_data['dn'][idx]]['establish'].append(demand_data['establish'][idx])
            ep_info[demand_data['sn'][idx]]['index'].append(demand_data['index'][idx])
            ep_info[demand_data['dn'][idx]]['index'].append(demand_data['index'][idx])
            ep_info[demand_data['sn'][idx]]['sn'].append(demand_data['sn'][idx])
            ep_info[demand_data['sn'][idx]]['dn'].append(demand_data['dn'][idx])
            ep_info[demand_data['dn'][idx]]['sn'].append(demand_data['sn'][idx])
            ep_info[demand_data['dn'][idx]]['dn'].append(demand_data['dn'][idx])
        else:
            # already added this flow
            pass

    return ep_info


def adjust_demand_load_to_ep_link_margin(demand_data, 
                                           new_num_demands,
                                           interarrival_time_dist,
                                           eps,
                                           node_dist,
                                           flow_size_dist,
                                           network_load_config,
                                           ep_link_margin=0.95,
                                           increment_factor=1.001,
                                           print_data=False):
    '''
    Decrease ep link loads of each ep link until <= ep link margin load (e.g. 0.95).
    If after this decrease the overall load is below target load, increase lowest
    ep link loads until get to target load. Therefore as target load tends to 1,
    node load distribution tends towards uniform distribution (since load on all
    eps will tend to 1) (however, flow size and node pair probability distributions
    remain unchanged).
    '''
    # load_rate = get_flow_centric_demand_data_load_rate(demand_data, method='mean_per_ep', eps=eps)
    load_rate = get_flow_centric_demand_data_overall_load_rate(demand_data, bidirectional_links=True)
    load_fraction = load_rate / network_load_config['network_rate_capacity']
    target_load_fraction = network_load_config['target_load_fraction']
    network_rate_capacity = network_load_config['network_rate_capacity']
    
    assert target_load_fraction <= 1, \
        'Must have target load fraction <= 1 for compatability with network rate capacity.'
    
    target_load_rate = network_rate_capacity*target_load_fraction
    init_load_rate = copy.deepcopy(load_rate)
    init_load_fraction = init_load_rate / network_rate_capacity

    # group ep info from demand data
    ep_info = group_demand_data_into_ep_info(demand_data, eps)

    # decrease ep link load to below ep link margin for all ep links which exceed it
    adjusted_eps = []
    for ep in eps:
        # total_info = sum(ep_info[ep]['flow_size'])
        # time_first_flow_arrived = min(ep_info[ep]['event_time'])
        # time_last_flow_arrived = max(ep_info[ep]['event_time'])
        # ep_load_frac = (total_info / (time_last_flow_arrived-time_first_flow_arrived)) / network_load_config['ep_link_capacity']
        ep_load_rate = get_flow_centric_demand_data_ep_load_rate(demand_data, ep, eps)
        ep_load_frac = ep_load_rate / network_load_config['ep_link_capacity']
        if ep_load_frac > ep_link_margin:
            adjusted_eps.append(ep)
            while ep_load_frac > ep_link_margin:
                # must decrease load by spreading out arrival times
                for i in range(len(ep_info[ep]['demand_data_idx'])):
                    demand_data['event_time'][ep_info[ep]['demand_data_idx'][i]] *= increment_factor
                    ep_info[ep]['event_time'][i] = demand_data['event_time'][ep_info[ep]['demand_data_idx'][i]]
                # time_first_flow_arrived = min(ep_info[ep]['event_time'])
                # time_last_flow_arrived = max(ep_info[ep]['event_time'])
                # ep_load_frac = (total_info / (time_last_flow_arrived-time_first_flow_arrived)) / network_load_config['ep_link_capacity']
                ep_load_rate = get_flow_centric_demand_data_ep_load_rate(demand_data, ep, eps)
                ep_load_frac = ep_load_rate / network_load_config['ep_link_capacity']

    # if overall load now below target load, increase loads of other ep links until get to target load
    # load_rate = get_flow_centric_demand_data_load_rate(demand_data, method='mean_per_ep', eps=eps)
    # load_frac = load_rate / network_rate_capacity
    load_rate = get_flow_centric_demand_data_overall_load_rate(demand_data, bidirectional_links=True)
    load_frac = load_rate / network_rate_capacity
    while load_frac < 0.99 * target_load_fraction:
        if len(adjusted_eps) == len(eps):
            raise Exception('All eps have been adjusted to be <= {} load margin, but load frac is still {} (target {}). Change target load, or change distributions and/or topology to be valid for your desired target load (e.g. node distribution might be too heavily skewed).'.format(ep_link_margin, load_frac, target_load_fraction))

        # DEBUG
        ep_loads = {ep: None for ep in eps}
        ep_last_event_times = {ep: None for ep in eps}
        ep_first_event_times = {ep: None for ep in eps}
        # print('Adjusted eps: {}'.format(adjusted_eps))

        # NEW
        for ep in ep_info.keys():
            ep_last_event_time = max(ep_info[ep]['event_time'])
            ep_last_event_times[ep] = ep_last_event_time
        ep_with_last_event = max(ep_last_event_times, key=ep_last_event_times.get)

        for ep in ep_info.keys():
            
            # DEBUG
            ep_load_rate = get_flow_centric_demand_data_ep_load_rate(demand_data, ep, eps)
            ep_load_frac = ep_load_rate / network_load_config['ep_link_capacity']
            first_event_time = min(ep_info[ep]['event_time'])
            ep_loads[ep] = ep_load_frac
            ep_first_event_times[ep] = first_event_time

            # update
            last_event_time = max(ep_info[ep]['event_time'])
            ep_last_event_times[ep] = last_event_time
            ep_with_last_event = max(ep_last_event_times, key=ep_last_event_times.get)

            if ep in adjusted_eps:
                # already adjusted this ep to <= ep link margin
                pass
            else:
                # check that not about to exceed ep link rate
                # time_last_flow_arrived = max(ep_info[ep]['event_time'])
                # time_first_flow_arrived = min(ep_info[ep]['event_time'])
                # total_info = sum(ep_info[ep]['flow_size'])
                # ep_load_frac = (total_info / (time_last_flow_arrived-time_first_flow_arrived)) / network_load_config['ep_link_capacity']
                ep_load_rate = get_flow_centric_demand_data_ep_load_rate(demand_data, ep, eps)
                ep_load_frac = ep_load_rate / network_load_config['ep_link_capacity']
                if ep_load_frac >= 0.99*ep_link_margin:
                    # can no longer increase load on this ep
                    adjusted_eps.append(ep)
                else:
                    # can try increase load on this ep to reach overall target load
                    found_flow_to_adjust = False 
                    for i in range(len(ep_info[ep]['demand_data_idx'])):
                        sn, dn = ep_info[ep]['sn'][i], ep_info[ep]['dn'][i]
                        if sn not in adjusted_eps and dn not in adjusted_eps:
                            found_flow_to_adjust = True

                            # # New
                            # demand_data['event_time'][ep_info[ep]['demand_data_idx'][i]] *= (1-(increment_factor-1))
                            # ep_info[ep]['event_time'][i] = demand_data['event_time'][ep_info[ep]['demand_data_idx'][i]]

                            # Old
                            if ep_with_last_event in adjusted_eps:
                                # cannot change overall load by adjusting total duration since load-limiting ep already at max load, must instead change total info
                                demand_data['flow_size'][ep_info[ep]['demand_data_idx'][i]] *= increment_factor
                                # ensure is integer
                                demand_data['flow_size'][ep_info[ep]['demand_data_idx'][i]] = int(demand_data['flow_size'][ep_info[ep]['demand_data_idx'][i]])
                                ep_info[ep]['flow_size'][i] = demand_data['flow_size'][ep_info[ep]['demand_data_idx'][i]]
                            else:
                                # can change overall load by adjusting total duration since load limiting ep not already at max load
                                demand_data['event_time'][ep_info[ep]['demand_data_idx'][i]] *= (1-(increment_factor-1))
                                ep_info[ep]['event_time'][i] = demand_data['event_time'][ep_info[ep]['demand_data_idx'][i]]
                    if not found_flow_to_adjust:
                        raise Exception('Adjusted ep loads as much as possible, but could only reach overall load {} (target {}). Either increase ep link margin (currently {}), decrease target load, or change distributions and/or topology to be valid for your requested overall load (e.g. node distributions might be too heavily skewed).'.format(load_frac, target_load_fraction, ep_link_margin))
        # load_rate = get_flow_centric_demand_data_load_rate(demand_data, method='mean_per_ep', eps=eps)
        # load_frac = load_rate / network_rate_capacity
        load_rate = get_flow_centric_demand_data_overall_load_rate(demand_data, bidirectional_links=True)
        load_frac = load_rate / network_rate_capacity
        print('Overall load frac: {} | Target: {}'.format(load_frac, target_load_fraction))
        # print('ep loads: {}\nep first event times: {}\nep last event times: {}\nep with last event:{}'.format(ep_loads, ep_first_event_times, ep_last_event_times, max(ep_last_event_times, key=ep_last_event_times.get)))
        

    if print_data:
        print('Final load rate | frac after adjusting ep loads to <= {}: {} | {}'.format(ep_link_margin, load_rate, load_frac))

    return demand_data


    



def increase_demand_load_to_target(demand_data, 
                                   num_demands, 
                                   interarrival_time_dist, 
                                   eps, 
                                   node_dist, 
                                   flow_size_dist, 
                                   network_load_config, 
                                   increment_factor=0.5,
                                   print_data=False):
    # load_rate = get_flow_centric_demand_data_load_rate(demand_data, method='mean_per_ep', eps=eps)
    # load_fraction = load_rate / network_load_config['network_rate_capacity']
    load_rate = get_flow_centric_demand_data_overall_load_rate(demand_data, bidirectional_links=True)
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
        # load_rate = get_flow_centric_demand_data_load_rate(demand_data, method='mean_per_ep', eps=eps)
        # load_fraction = load_rate / network_load_config['network_rate_capacity']
        load_rate = get_flow_centric_demand_data_overall_load_rate(demand_data, bidirectional_links=True)
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
    # load_rate = get_flow_centric_demand_data_load_rate(demand_data, method='mean_per_ep', eps=eps)
    load_rate = get_flow_centric_demand_data_overall_load_rate(demand_data, bidirectional_links=True)
    load_fraction = load_rate / network_load_config['network_rate_capacity']
    target_load_fraction = network_load_config['target_load_fraction']
    network_rate_capacity = network_load_config['network_rate_capacity']
    
    assert target_load_fraction <= 1, \
        'Must have target load fraction <= 1 for compatability with network rate capacity.'
    
    target_load_rate = network_rate_capacity*target_load_fraction
    init_load_rate = copy.deepcopy(load_rate)
    init_load_fraction = init_load_rate / network_rate_capacity

    if load_rate > target_load_rate:
        # increase interarrival time dist until get target load rate
        num_loops = 1
        while load_rate > target_load_rate:
            # adjust interarrival dist by adjusting event times
            demand_data['event_time'] *= increment_factor
            new_interarrival_time_dist = {}
            # for rand_var, prob in interarrival_time_dist.items():
                # new_rand_var = rand_var * increment_factor
                # new_interarrival_time_dist[new_rand_var] = prob

            # # update interarrival time dist
            # interarrival_time_dist = new_interarrival_time_dist

            # # re-create
            # demand_data = create_flow_centric_demand_data(num_demands=num_demands,
                                                                      # eps=eps,
                                                                      # node_dist=node_dist,
                                                                      # flow_size_dist=flow_size_dist,
                                                                      # interarrival_time_dist=interarrival_time_dist,
                                                                      # print_data=print_data)

            # load_rate = get_flow_centric_demand_data_load_rate(demand_data, method='mean_per_ep', eps=eps)
            load_rate = get_flow_centric_demand_data_overall_load_rate(demand_data, bidirectional_links=True)

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

def get_flow_centric_demand_data_ep_load_rate(demand_data, ep, eps, method='all_eps'):
    '''
    If method=='all_eps', duration is time_last_flow_arrived-time_first_flow_arrived
    across all endpoints. If method=='per_ep', duration is time_last_flow_arrived-time_first_flow_arrived
    for this specific ep.
    '''
    ep_info = group_demand_data_into_ep_info(demand_data, eps)
    total_info = sum(ep_info[ep]['flow_size'])
    if method == 'per_ep':
        time_first_flow_arrived = min(ep_info[ep]['event_time'])
        time_last_flow_arrived = max(ep_info[ep]['event_time'])
    elif method == 'all_eps':
        time_first_flow_arrived = min(demand_data['event_time'])
        time_last_flow_arrived = max(demand_data['event_time'])
    duration = time_last_flow_arrived - time_first_flow_arrived
    load_rate = total_info / duration
    
    return load_rate

def get_flow_centric_demand_data_overall_load_rate(demand_data, bidirectional_links=True):
    '''
    If flow connections are bidirectional_links, 1 flow takes up 2 endpoint links (the
    source link and the destination link), therefore effecitvely takes up load rate
    2*flow_size*duration bandwidth. If not bidriectional, only takes up
    1*flow_size*duration since only occupies bandwidth for 1 of these links.

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

    if bidirectional_links:
        # 1 flow occupies 2 endpoint links therefore has 2*flow_size load
        load_rate = 2*info_arrived/duration
    else:
        load_rate = info_arrived/duration

    return load_rate


# def get_flow_centric_demand_data_load_rate(demand_data, method='mean_all_eps', bidirectional_links=True, **kwargs):
    # '''
    # If flow connections are bidirectional_links, 1 flow takes up 2 endpoint links (the
    # source link and the destination link), therefore effecitvely takes up load rate
    # 2*flow_size*duration bandwidth. If not bidriectional, only takes up
    # 1*flow_size*duration since only occupies bandwidth for 1 of these links.

    # If method == 'mean_per_ep', will calculate the total network load as being the mean
    # average load on each endpoint link (i.e. sum info requests for each link ->
    # find load of each link -> find mean of ep link loads)

    # If method == 'mean_all_eps', will calculate the total network load as being
    # the average load over all endpoint links (i.e. sum info requests for all links
    # -> find overall load of network)
    # '''
    # info_arrived = get_flow_centric_demand_data_total_info_arrived(demand_data)
    # first_event_time, last_event_time = get_first_last_flow_arrival_times(demand_data)
    # duration = last_event_time - first_event_time

    # if method == 'mean_per_ep':
        # ep_loads = {ep: 0 for ep in kwargs['eps']}
        # ep_info = group_demand_data_into_ep_info(demand_data, kwargs['eps'])
        # for ep in kwargs['eps']:
            # total_info = sum(ep_info[ep]['flow_size'])
            # # time_first_flow_arrived = min(ep_info[ep]['event_time'])
            # # time_last_flow_arrived = max(ep_info[ep]['event_time'])
            # # duration = time_last_flow_arrived - time_first_flow_arrived
            # ep_loads[ep] = total_info / duration
        # load_rate = np.mean(list(ep_loads.values())) * len(kwargs['eps'])

    # elif method == 'mean_all_eps':
        # if bidirectional_links:
            # # 1 flow occupies 2 endpoint links therefore has 2*flow_size load
            # load_rate = 2*info_arrived/duration
        # else:
            # load_rate = info_arrived/duration

    # else:
        # raise Exception('Unrecognised load rate calculation method {}'.format(method))
    
    # return load_rate



def get_flow_centric_demand_data_total_info_arrived(demand_data): 
    info_arrived = 0
    # print('flow size {} {}'.format(type(demand_data['flow_size']), type(demand_data['flow_size'][0])))
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


































