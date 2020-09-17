'''Module for generating node distributions.'''

from trafpy.generator.src.dists import plot_dists
from trafpy.generator.src.dists import val_dists
from trafpy.generator.src import tools

import sys
import numpy as np
import time
import json
import copy




def assign_probs_to_matrix(eps, probs, matrix=None):
    '''Assigns probabilities to 2D matrix.

    N.B. Assumes probs are given in order of matrix indices when looping
    for src in eps for dst in eps

    '''
    if matrix is None:
        matrix = np.zeros((len(eps),len(eps)))
    num_nodes, num_pairs, node_to_index, index_to_node = tools.get_network_params(eps)
    
    iter = np.nditer(np.array(probs))
    for src in eps:
        for dst in eps:
            if src == dst:
                continue
            elif src > dst:
                # making symmetric so skip this side of diagonal
                continue
            else:
                prob = next(iter)
                src_idx = node_to_index[src]
                dst_idx = node_to_index[dst]
                matrix[src_idx,dst_idx] = prob
                matrix[dst_idx,src_idx] = prob

    return matrix


def gen_uniform_node_dist(eps, 
                          rack_prob_config=None,
                          path_to_save=None, 
                          plot_fig=False, 
                          show_fig=False,
                          print_data=False):
    '''Generates a uniform node distribution.

    Args:
        eps (list): List of network node endpoints that can act as sources
            & destinations
        rack_prob_config (dict): Network endpoints/servers are often grouped into
            physically local clusters or `racks'. Different networks may have 
            different levels of inter- (between) and intra- (within) rack communication.
            If rack_prob_config is left as None, will assume that server-server
            srs-dst requests are independent of which rack they might be in.
            If specified, dict should have a `racks_dict' key, whose value is a dict
            with keys as rack labels (e.g. 'rack_0', 'rack_1' etc.) and whose value
            for each key is a list of the endpoints in the respective rack
            (e.g. [`server_0', `server_24', `server_56', ...]), and a `prob_inter_rack'
            key whose value is a float (e.g. 0.9), setting the probability
            that a chosen src endpoint has a destination which is outside of its rack.
            If you want to e.g. designate an entire rack as a 'hot rack' (many
            traffic requests occur from this rack), would specify skewed_nodes to
            contain the list of servers in this rack and configure rack_prob_config
            appropriately.
        path_to_save (str): Path to directory (with file name included) in which
            to save generated distribution. E.g. path_to_save='data/dists/my_dist'.
        plot_fig (bool): Whether or not to plot fig. If True, will return fig.
        show_fig (bool): Whether or not to plot and show fig. If True, will
            return and display fig.
        print_data (bool): Whether or not to print extra information about the
            generated data.
    
    Returns:
        tuple: Tuple containing:
            - **node_dist** (*numpy array*): 2D matrix array of souce-destination pair
              probabilities of being chosen.
            - **fig** (*matplotlib.figure.Figure, optional*): Node distributions
              plotted as a 2D matrix. To return, set show_fig=True and/or plot_fig=True.

    '''
    # init network params
    num_nodes, num_pairs, node_to_index, index_to_node = tools.get_network_params(eps)
    node_dist = np.zeros((num_nodes, num_nodes))
    
    # uniform prob each pair chosen
    prob_pair_chosen = (np.ones((num_pairs))/((num_pairs)))/2
    if print_data:
        print('Prob pair chosen:\n{}'.format(prob_pair_chosen))

    # assign probabilites to matrix
    node_dist = assign_probs_to_matrix(eps=eps,
                                       probs=prob_pair_chosen,
                                       matrix=node_dist)

    if rack_prob_config is not None:
        # adjust node prob dist to account for rack prob config
        node_dist = adjust_node_dist_for_rack_prob_config(rack_prob_config,
                                                          eps,
                                                          node_dist,
                                                          print_data=print_data)

    matrix_sum = np.round(np.sum(node_dist),2)

    assert matrix_sum == 1, \
        'matrix must sum to 1, but is {}'.format(matrix_sum)
    
    if print_data:
        print('Normalised matrix:\n{}'.format(node_dist))
        print('Normalised matrix sum: {}'.format(matrix_sum))
    if path_to_save is not None:
        tools.pickle_data(path, node_dist)
    if plot_fig or show_fig:
        fig = plot_dists.plot_node_dist(node_dist=node_dist, 
                                        eps=eps,
                                        show_fig=show_fig)
        return node_dist, fig

    else:
        return node_dist





def gen_uniform_multinomial_exp_node_dist(eps,
                                          rack_prob_config=None,
                                          path_to_save=None,
                                          plot_fig=False,
                                          show_fig=False,
                                          print_data=False):
    '''Runs multinomial exp with uniform initial probability to generate slight skew.

    Runs a multinomial experiment where each node pair has same (uniform)
    probability of being chosen. Will generate a node demand distribution
    where a few pairs & nodes have a slight skew in demand

    Args:
        eps (list): List of network node endpoints that can act as sources
            & destinations
        rack_prob_config (dict): Network endpoints/servers are often grouped into
            physically local clusters or `racks'. Different networks may have 
            different levels of inter- (between) and intra- (within) rack communication.
            If rack_prob_config is left as None, will assume that server-server
            srs-dst requests are independent of which rack they might be in.
            If specified, dict should have a `racks_dict' key, whose value is a dict
            with keys as rack labels (e.g. 'rack_0', 'rack_1' etc.) and whose value
            for each key is a list of the endpoints in the respective rack
            (e.g. [`server_0', `server_24', `server_56', ...]), and a `prob_inter_rack'
            key whose value is a float (e.g. 0.9), setting the probability
            that a chosen src endpoint has a destination which is outside of its rack.
            If you want to e.g. designate an entire rack as a 'hot rack' (many
            traffic requests occur from this rack), would specify skewed_nodes to
            contain the list of servers in this rack and configure rack_prob_config
            appropriately.
        path_to_save (str): Path to directory (with file name included) in which
            to save generated distribution. E.g. path_to_save='data/dists/my_dist'.
        plot_fig (bool): Whether or not to plot fig. If True, will return fig.
        show_fig (bool): Whether or not to plot and show fig. If True, will
            return and display fig.
        print_data (bool): Whether or not to print extra information about the
            generated data.
    
    Returns:
        tuple: Tuple containing:
            - **node_dist** (*numpy array*): 2D matrix array of souce-destination pair
              probabilities of being chosen.
            - **fig** (*matplotlib.figure.figure, optional*): node distribution
              plotted as a 2d matrix. to return, set show_fig=true and/or plot_fig=true.

    '''
    # initialise graph params
    num_nodes, num_pairs, node_to_index, index_to_node = tools.get_network_params(eps)
    node_dist = np.zeros((num_nodes, num_nodes))
    prob_pair_chosen = np.ones((num_pairs))/((num_pairs))

    # run multinomial exp to get no. times each pair chosen
    counter_array = np.random.multinomial(500, prob_pair_chosen, size=1)[0]

    # get probabilities each pair chosen
    counter_array_prob_dist = counter_array/1000

    # assign probabilites to matrix
    node_dist = assign_probs_to_matrix(eps=eps,
                                       probs=counter_array_prob_dist,
                                       matrix=node_dist)

    if rack_prob_config is not None:
        # adjust node prob dist to account for rack prob config
        node_dist = adjust_node_dist_for_rack_prob_config(rack_prob_config,
                                                          eps,
                                                          node_dist,
                                                          print_data=print_data)

    matrix_sum = np.round(np.sum(node_dist),2)

    assert matrix_sum == 1, \
        'matrix must sum to 1, but is {}'.format(matrix_sum)
    
    if print_data:
        print('Normalised matrix:\n{}'.format(node_dist))
        print('Normalised matrix sum: {}'.format(matrix_sum))
    if path_to_save is not None:
        tools.pickle_data(path, node_dist)
    if plot_fig or show_fig:
        fig = plot_dists.plot_node_dist(node_dist=node_dist, 
                                        eps=eps,
                                        show_fig=show_fig)
        return node_dist, fig

    else:
        return node_dist


def gen_multimodal_node_dist(eps,
                             skewed_nodes=[],
                             skewed_node_probs=[],
                             num_skewed_nodes=None,
                             rack_prob_config=None,
                             path_to_save=None,
                             plot_fig=False,
                             show_fig=False,
                             print_data=False):
    '''Generates a multimodal node distribution.

    Generates a multimodal node demand distribution i.e. certain nodes
    have a certain specified probability of being chosen. If no
    skewed nodes given, randomly selects random no. node(s) to skew. If no 
    skew node probabilities given, random selects probability with which
    to skew the node between 0.5 and 0.8. If no num skewed nodes given,
    randomly chooses number of nodes to skew.

    Args:
        eps (list): List of network node endpoints that can act as sources
            & destinations
        skewed_nodes (list): Node(s) to whose probability of being
            chosen you want to skew/specify
        skewed_node_probs (list): Probabilit(y)(ies) of node(s)
            being chosen/specified skews
        num_skewed_nodes (int): Number of nodes to skew. If None, will gen
            a number between 10% and 30% of the total number of nodes in network
        rack_prob_config (dict): Network endpoints/servers are often grouped into
            physically local clusters or `racks'. Different networks may have 
            different levels of inter- (between) and intra- (within) rack communication.
            If rack_prob_config is left as None, will assume that server-server
            srs-dst requests are independent of which rack they might be in.
            If specified, dict should have a `racks_dict' key, whose value is a dict
            with keys as rack labels (e.g. 'rack_0', 'rack_1' etc.) and whose value
            for each key is a list of the endpoints in the respective rack
            (e.g. [`server_0', `server_24', `server_56', ...]), and a `prob_inter_rack'
            key whose value is a float (e.g. 0.9), setting the probability
            that a chosen src endpoint has a destination which is outside of its rack.
            If you want to e.g. designate an entire rack as a 'hot rack' (many
            traffic requests occur from this rack), would specify skewed_nodes to
            contain the list of servers in this rack and configure rack_prob_config
            appropriately.
        path_to_save (str): Path to directory (with file name included) in which
            to save generated distribution. E.g. path_to_save='data/dists/my_dist'.
        plot_fig (bool): Whether or not to plot fig. If True, will return fig.
        show_fig (bool): Whether or not to plot and show fig. If True, will
            return and display fig.
        print_data (bool): Whether or not to print extra information about the
            generated data.
    
    Returns:
        tuple: Tuple containing:
            - **node_dist** (*numpy array*): 2D matrix array of souce-destination pair
              probabilities of being chosen.
            - **fig** (*matplotlib.figure.Figure, optional*): Node distributions
              plotted as a 2D matrix. To return, set show_fig=True and/or plot_fig=True.
    
    ''' 
    # initialise graph params
    num_nodes, num_pairs, node_to_index, index_to_node = tools.get_network_params(eps)
    node_dist = np.zeros((num_nodes, num_nodes))
    
    if num_skewed_nodes is None:
        num_skewed_nodes = len(skewed_nodes)

    if len(skewed_nodes) == 0:
        if num_skewed_nodes == 0:
            # randomly choose number of nodes to skew
            min_skews = int(0.1*num_nodes)
            max_skews = int(0.3*num_nodes)
            if min_skews < 1:
                min_skews = 1
                max_skews = 2
            num_skewed_nodes = np.random.randint(min_skews, max_skews+1)
        # randomly choose a node
        skewed_nodes = list(np.random.choice(eps, size=num_skewed_nodes))

    if len(skewed_node_probs) == 0:
        # randomly choose skew between range
        min_prob = 0.5/num_skewed_nodes
        max_prob = 0.8/num_skewed_nodes
        skewed_node_probs = np.random.uniform(min_prob, 
                                              max_prob, 
                                              size=num_skewed_nodes)
        skewed_node_probs = list(np.round(skewed_node_probs,2))
    if print_data:
        print('Num skewed nodes: {}'.format(num_skewed_nodes))
        print('Chosen for skew:\n{}'.format(skewed_nodes))
        print('Chosen probs:\n{}'.format(skewed_node_probs))

    # get indices of node pairs to skew
    pairs_per_node = num_nodes - 1
    skewed_node_pair_indices = np.zeros((pairs_per_node, num_skewed_nodes))
    for skewed_node_iter in range(num_skewed_nodes):
        skewed_node_pair_iter = 0
        matrix_iter = 0
        for src in eps:
            for dst in eps:
                if src == dst:
                    continue
                elif src > dst:
                    continue
                else:
                    skewed_node = skewed_nodes[skewed_node_iter]
                    if skewed_node == src or skewed_node == dst:
                        skewed_node_pair_indices[skewed_node_pair_iter,skewed_node_iter] = matrix_iter
                        skewed_node_pair_iter+=1
                    matrix_iter += 1
    if print_data:
        print('Skewed node pair indices:\n{}'.format(skewed_node_pair_indices))

    # find prob of each skewed node pair being chosen
    probs_per_skewed_pair = np.zeros(num_skewed_nodes)
    for node in range(num_skewed_nodes):
        probs_per_skewed_pair[node] = skewed_node_probs[node] / pairs_per_node

    # update prob pair chosen for each pair with a skewed node 
    prob_pair_chosen = np.zeros(num_pairs)
    iter = np.nditer(skewed_node_pair_indices)
    for skewed_node_iter in range(num_skewed_nodes):
        for skewed_node_pair_iter in range(pairs_per_node):
            for pair in range(num_pairs):
                if pair == skewed_node_pair_indices[skewed_node_pair_iter,skewed_node_iter]:
                    # add to skew of node
                    prob_pair_chosen[pair] += probs_per_skewed_pair[skewed_node_iter]
                else:
                    continue 
    # will allocate twice
    prob_pair_chosen = prob_pair_chosen / 2
    total_skew_prob = np.sum(prob_pair_chosen)

    # assign prob pair chosen to any pairs w/o skewed nodes
    if total_skew_prob < 0.5:
        remaining_pairs = np.count_nonzero(prob_pair_chosen == 0)
        prob_dist = np.ones((remaining_pairs))/((remaining_pairs))
        counter_array = np.random.multinomial(500,
                                              prob_dist,
                                              size=1)[0]
        counter_array_prob_dist = (counter_array/1000)
        counter_array_prob_dist = ((0.5-total_skew_prob)/0.5) * counter_array_prob_dist
        iter = np.nditer(counter_array_prob_dist)
        for pair in range(len(prob_pair_chosen)):
            if prob_pair_chosen[pair] == 0:
                prob_pair_chosen[pair] = next(iter)
    if print_data:
        print('Prob pair chosen:\n{}'.format(prob_pair_chosen))

    # assign probabilites to normalised demand matrix
    node_dist = assign_probs_to_matrix(eps=eps,
                                       probs=prob_pair_chosen,
                                       matrix=node_dist)

    if rack_prob_config is not None:
        # adjust node prob dist to account for rack prob config
        node_dist = adjust_node_dist_for_rack_prob_config(rack_prob_config,
                                                          eps,
                                                          node_dist,
                                                          print_data=print_data)

    matrix_sum = np.round(np.sum(node_dist),2)
    assert matrix_sum == 1, \
        'matrix must sum to 1, but is {}'.format(matrix_sum)
    
    if print_data:
        print('Normalised matrix:\n{}'.format(node_dist))
        print('Normalised matrix sum: {}'.format(matrix_sum))
    if path_to_save is not None:
        tools.pickle_data(path, node_dist)
    if plot_fig or show_fig:
        fig = plot_dists.plot_node_dist(node_dist=node_dist, 
                                        eps=eps,
                                        show_fig=show_fig)
        return node_dist, fig

    else:
        return node_dist

def get_suitable_destination_node_for_rack_config(sn, node_dist, eps, ep_to_rack, rack_to_ep, inter_rack):
    '''Given source node, finds destination node given inter and intra rack config.'''
    sn_rack = ep_to_rack[sn]

    # get list of suitable destination nodes
    dn_eps = []
    if inter_rack:
        # get list of inter rack destination nodes for this source node
        for rack in rack_to_ep.keys():
            if rack != sn_rack:
                rack_eps = rack_to_ep[rack]
                dn_eps.append(rack_eps)
    else:
        # get list of intra rack destination nodes for this source node
        for rack in rack_to_ep.keys():
            if rack == sn_rack:
                rack_eps = rack_to_ep[rack]
                dn_eps.append(rack_eps)
    # flatten
    dn_eps_flat = []
    for l in dn_eps:
        for d in l:
            dn_eps_flat.append(d)

    # find suitable destination for source node
    dn = copy.deepcopy(sn)
    while sn == dn:
        dn = gen_demand_nodes(eps=dn_eps_flat,
                              node_dist=node_dist,
                              size=1,
                              axis=1)[0]

    return dn


def adjust_node_dist_for_rack_prob_config(rack_prob_config,
                                          eps,
                                          node_dist,
                                          num_exps_factor=100,
                                          print_data=False):
    '''Takes node dist and uses it to generate new node dist given inter- and intra-rack configuration.

    Different DCNs have different inter and intra rack traffic. This function
    allows you to specify how much of your traffic should be inter and intra rack.

    Args:
        rack_prob_config (dict): Network endpoints/servers are often grouped into
            physically local clusters or `racks'. Different networks may have 
            different levels of inter- (between) and intra- (within) rack communication.
            If rack_prob_config is left as None, will assume that server-server
            srs-dst requests are independent of which rack they might be in.
            If specified, dict should have a `racks_dict' key, whose value is a dict
            with keys as rack labels (e.g. 'rack_0', 'rack_1' etc.) and whose value
            for each key is a list of the endpoints in the respective rack
            (e.g. [`server_0', `server_24', `server_56', ...]), and a `prob_inter_rack'
            key whose value is a float (e.g. 0.9), setting the probability
            that a chosen src endpoint has a destination which is outside of its rack.
            If you want to e.g. designate an entire rack as a 'hot rack' (many
            traffic requests occur from this rack), would specify skewed_nodes to
            contain the list of servers in this rack and configure rack_prob_config
            appropriately.
        eps (list): List of network node endpoints that can act as sources
            & destinations.
        node_dist (numpy array): 2D matrix array of source-destination pair
            probabilities of being chosen.
        num_exps_factor (int): Factor by which to multiply number of eps to get
            the number of multinomial experiments to run when generating new
            node dist.
        print_data (bool): Whether or not to print extra information about the
            generated data.

    '''

    node_dist = copy.deepcopy(node_dist)

    # switch racks_dict keys and values to make hashing easier
    racks_dict = {}
    for key, val in rack_prob_config['racks_dict'].items():
        for v in val:
            if v not in racks_dict.keys():
                racks_dict[v] = key

    # run multinomial exp to incpororate rack probabilities
    num_experiments = num_exps_factor * len(eps) 
    sampled_pairs = {}
    inter_counter = 0
    intra_counter = 0
    for exp in range(num_experiments):
        # sample if connection should be intra or inter rack
        inter_rack = np.random.choice(a=[True, False], 
                                      p=[rack_prob_config['prob_inter_rack'], 
                                         1-rack_prob_config['prob_inter_rack']])
        if inter_rack:
            inter_counter += 1
        else:
            intra_counter += 1

        # sample a source node
        sn = gen_demand_nodes(eps=eps,
                              node_dist=node_dist,
                              size=1,
                              axis=0)[0]

        # sample destination node given inter_rack config
        dn = get_suitable_destination_node_for_rack_config(sn, 
                                                           copy.deepcopy(node_dist), 
                                                           eps, 
                                                           ep_to_rack=racks_dict, 
                                                           rack_to_ep=rack_prob_config['racks_dict'], 
                                                           inter_rack=inter_rack)

        pair = json.dumps([sn, dn])
        if pair not in sampled_pairs:
            # check if switched src-dst pair that already occurred
            pair_switched = json.dumps([dn, sn])
            if pair_switched not in sampled_pairs:
                sampled_pairs[pair] = 1 # init first occurrence of pair
            else:
                # pair already seen before
                sampled_pairs[pair_switched] += 1
        else:
            # pair already seen before
            sampled_pairs[pair] += 1

    # convert sampled pairs dict to rand vars
    sampled_pairs = val_dists.convert_key_occurrences_to_data(list(sampled_pairs.keys()), list(sampled_pairs.values()))

    if print_data:
        total = inter_counter + intra_counter
        print('Number inter-rack requests: {} ({}%)'.format(inter_counter, inter_counter*100/total))
        print('Number intra-rack requests: {} ({}%)'.format(intra_counter, intra_counter*100/total))

    # convert sampled pairs list into prob dist
    unique_vals, pmf = val_dists.gen_discrete_prob_dist(sampled_pairs)
    sampled_pair_prob_dist = {unique_var: prob for unique_var, prob in zip(unique_vals, pmf)}

    # insert 0 probabilites for any pairs that were never sampled in multinomial exp
    num_nodes, num_pairs, node_to_index, index_to_node = tools.get_network_params(eps)
    index_to_pair, pair_to_index = get_network_pair_mapper(eps)
    if num_pairs != len(sampled_pair_prob_dist.keys()):
        # some pairs were never chosen, find which are missing
        chosen_pair_indices = {}
        for pair in sampled_pair_prob_dist.keys():
            try:
                chosen_pair_indices[pair_to_index[pair]] = None
            except KeyError:
                pair = json.loads(pair)
                pair = [pair[1],pair[0]]
                pair = json.dumps(pair)
                chosen_pair_indices[pair_to_index[pair]] = None

        for index in index_to_pair.keys():
            if index not in chosen_pair_indices:
                # index is missing, update sampled pairs dict with 0 prob for this index
                sampled_pair_prob_dist[json.dumps(index_to_pair[index])] = 0

    # convert prob dist into dict whose keys are node dist matrix indices
    index_to_pair, pair_to_index = get_network_pair_mapper(eps)
    matrix_pair_prob_dist = {}
    for key in sampled_pair_prob_dist.keys():
        try:
            matrix_index = pair_to_index[key]
        except KeyError:
            # switch src and dst
            pair = json.loads(key)
            key = json.dumps([pair[1], pair[0]])
            matrix_index = pair_to_index[key]
        try:
            matrix_pair_prob_dist[matrix_index] = sampled_pair_prob_dist[key]
        except KeyError:
            # switch src and dst back for sampled pair prob dist
            pair = json.loads(key)
            key = json.dumps([pair[1], pair[0]])
            matrix_pair_prob_dist[matrix_index] = sampled_pair_prob_dist[key]

    # sort so that get correct ordering of pair probs when assign to node matrix
    sorted_matrix_pair_prob_dist_keys = sorted((matrix_pair_prob_dist.keys()))

    # generate new node matrix prob dist (which now accounts for rack probs)
    prob_pair_chosen = []
    for key in sorted_matrix_pair_prob_dist_keys:
        prob = matrix_pair_prob_dist[key]/2 # divide by 2 since one side of diagonal
        prob_pair_chosen.append(prob)


    if print_data:
        print('Prob pair chosen after accounting for rack prob config:\n{}'.format(prob_pair_chosen))
        print(len(prob_pair_chosen))





    node_dist = assign_probs_to_matrix(eps=eps,
                                       probs=prob_pair_chosen,
                                       matrix=node_dist)
    
    return node_dist





def get_network_pair_mapper(eps):
    '''Gets dicts mapping network endpoint indices to and from node dist matrix.'''
    num_nodes, num_pairs, node_to_index, index_to_node = tools.get_network_params(eps)

    index_to_pair = {}
    pair_to_index = {}
    pair_index = 0
    for src in eps:
        for dst in eps:
            if src == dst:
                continue
            elif src > dst:
                # making symmetric so skip this side of diagonal
                continue
            else:
                index_to_pair[pair_index] = [src, dst]
                pair_to_index[json.dumps([src, dst])] = pair_index
                pair_index += 1

    return index_to_pair, pair_to_index


def gen_multimodal_node_pair_dist(eps,
                                  skewed_pairs = [],
                                  skewed_pair_probs = [],
                                  num_skewed_pairs=None,
                                  rack_prob_config=None,
                                  path_to_save=None,
                                  plot_fig=False,
                                  show_fig=False,
                                  print_data=False):
    '''Generates a multimodal node pair distribution.

    Generates a multimodal node pair demand distribution i.e. certain node
    pairs have a certain specified probability of being chosen. If no
    skewed pairs given, randomly selects pair to skew. If no skew 
    pair probabilities given, random selects probability with which
    to skew the pair between 0.1 and 0.3. If no num skewed pairs given,
    randomly chooses number of pairs to skew.

    Args:
        eps (list): List of network node endpoints that can act as sources
            & destinations.
        skewed_pairs (list of lists): List of the node pairs [src,dst] to 
            skew.
        skewed_pair_probs (list): Probabilities of node pairs being
            chosen.
        num_skewed_pairs (int): Number of pairs to randomly skew.
        rack_prob_config (dict): Network endpoints/servers are often grouped into
            physically local clusters or `racks'. Different networks may have 
            different levels of inter- (between) and intra- (within) rack communication.
            If rack_prob_config is left as None, will assume that server-server
            srs-dst requests are independent of which rack they might be in.
            If specified, dict should have a `racks_dict' key, whose value is a dict
            with keys as rack labels (e.g. 'rack_0', 'rack_1' etc.) and whose value
            for each key is a list of the endpoints in the respective rack
            (e.g. [`server_0', `server_24', `server_56', ...]), and a `prob_inter_rack'
            key whose value is a float (e.g. 0.9), setting the probability
            that a chosen src endpoint has a destination which is outside of its rack.
            If you want to e.g. designate an entire rack as a 'hot rack' (many
            traffic requests occur from this rack), would specify skewed_nodes to
            contain the list of servers in this rack and configure rack_prob_config
            appropriately.
        path_to_save (str): Path to directory (with file name included) in which
            to save generated distribution. E.g. path_to_save='data/dists/my_dist'.
        plot_fig (bool): Whether or not to plot fig. If True, will return fig.
        show_fig (bool): Whether or not to plot and show fig. If True, will
            return and display fig.
        print_data (bool): Whether or not to print extra information about the
            generated data.
    
    Returns:
        tuple: Tuple containing:
            - **node_dist** (*numpy array*): 2D matrix array of souce-destination pair
              probabilities of being chosen.
            - **fig** (*matplotlib.figure.Figure, optional*): Node distributions
              plotted as a 2D matrix. To return, set show_fig=True and/or plot_fig=True.

    '''
    # initialise graph params
    num_nodes, num_pairs, node_to_index, index_to_node = tools.get_network_params(eps)
    node_dist = np.zeros((num_nodes, num_nodes))
    
    if num_skewed_pairs is None:
        num_skewed_pairs = len(skewed_pairs)

    if len(skewed_pairs) == 0:
        if num_skewed_pairs == 0:
            #randomly choose number of nodes to skew
            min_skews = int(0.1*num_pairs)
            max_skews = int(0.3*num_pairs)
            if min_skews < 1:
                min_skews = 1
                max_skews = 2
            num_skewed_pairs = np.random.randint(min_skews, max_skews+1)
        # randomly choose src and dst for pairs
        nodes = eps
        src = np.random.choice(nodes, size=num_skewed_pairs)
        dst = np.random.choice(nodes, size=num_skewed_pairs)
        # remove src-dst conflicts and repeated pairs
        for pair_iter in range(num_skewed_pairs):
            try:
                pair = skewed_pairs[pair_iter]
            except IndexError:
                pair = []
            while (skewed_pairs.count(pair) > 1 or  src[pair_iter] == dst[pair_iter]):
                # remove repeated pairs and src-dst conflicts
                print('dst: {}'.format(dst))
                print('pair iter: {}'.format(pair_iter))
                print('nodes: {}'.format(nodes))
                dst[pair_iter] = np.random.choice(nodes, size=1)[0]
            skewed_pairs.append([src[pair_iter],dst[pair_iter]])
        # keep src<dst convention consistent
        for pair_iter in range(num_skewed_pairs):
            pair = skewed_pairs[pair_iter]
            if pair[0] > pair[1]:
                # swap src and dst to keep convention consistent
                temp_src = pair[1]
                temp_dst = pair[0]
                skewed_pairs[pair_iter] = [temp_src,temp_dst]
        # update src and dst
        for pair in range(len(src)):
            src[pair] = skewed_pairs[pair][0]
            dst[pair] = skewed_pairs[pair][1]
    else:
        assert (any(isinstance(pair,list) for pair in skewed_pairs)), \
                'must enter skewed_pairs as list of lists'
        src = [pair[0] for pair in skewed_pairs]
        dst = [pair[1] for pair in skewed_pairs]
    
    if print_data:
        print('Chosen sources:\n{}'.format(src))
        print('Chosen destinations:\n{}'.format(dst))
        print('Num skewed pairs: {}'.format(num_skewed_pairs))
        print('Chosen pairs to skew:\n{}'.format(skewed_pairs))

    if len(skewed_pair_probs) == 0:
        # randomly choose skew between range
        min_prob = 0.3/num_skewed_pairs
        max_prob = 0.5/num_skewed_pairs
        skewed_pair_probs = np.random.uniform(min_prob, 
                                              max_prob, 
                                              size=num_skewed_pairs)
        skewed_pair_probs = list(np.round(skewed_pair_probs,2))
    if print_data:
        print('Skew probs:\n{}'.format(skewed_pair_probs))

    # get indices of node pairs to skew
    skewed_pair_indices = np.zeros(len(skewed_pairs))
    skewed_pair_iter = 0
    for pair in range(len(skewed_pairs)):
        matrix_iter = 0
        for src in eps:
            for dst in eps:
                if src == dst:
                    continue
                elif src > dst:
                    continue
                else:
                    if skewed_pairs[pair][0] == src and skewed_pairs[pair][1] == dst:
                        skewed_pair_indices[skewed_pair_iter] = matrix_iter
                        skewed_pair_iter += 1
                    elif skewed_pairs[pair][0] == dst and skewed_pairs[pair][1] == src:
                        skewed_pair_indices[skewed_pair_iter] = matrix_iter
                        skewed_pair_iter += 1
                    matrix_iter += 1
    if print_data:
        print('Skew indices:\n{}'.format(skewed_pair_indices))

    # update prob pair chosen for each skewed pair
    prob_pair_chosen = np.zeros(num_pairs)
    iter = np.nditer(np.asarray(skewed_pair_probs))
    for skewed_pair_iter in range(num_skewed_pairs):
        for pair_iter in range(num_pairs):
            if pair_iter == skewed_pair_indices[skewed_pair_iter]:
                # add to skew of pair
                prob = next(iter)
                prob_pair_chosen[pair_iter] += prob
            else:
                continue
    # will allocate twice
    prob_pair_chosen = prob_pair_chosen / 2
    total_skew_prob = np.sum(prob_pair_chosen)

    # assign prob pair chosen to any pairs w/o skew
    if total_skew_prob < 0.5:
        remaining_pairs = np.count_nonzero(prob_pair_chosen == 0)
        prob_dist = np.ones((remaining_pairs))/((remaining_pairs))
        counter_array = np.random.multinomial(500,
                                              prob_dist,
                                              size=1)[0]
        counter_array_prob_dist = (counter_array/1000)
        counter_array_prob_dist = ((0.5-total_skew_prob)/0.5) * counter_array_prob_dist
        iter = np.nditer(counter_array_prob_dist)
        for pair in range(len(prob_pair_chosen)):
            if prob_pair_chosen[pair] == 0:
                prob_pair_chosen[pair] = next(iter)
    if print_data:
        print('Prob pair chosen:\n{}'.format(prob_pair_chosen))

    # assign probabilites to normalised demand matrix
    node_dist = assign_probs_to_matrix(eps=eps,
                                       probs=prob_pair_chosen,
                                       matrix=node_dist)

    if rack_prob_config is not None:
        # adjust node prob dist to account for rack prob config
        node_dist = adjust_node_dist_for_rack_prob_config(rack_prob_config,
                                                          eps,
                                                          node_dist,
                                                          print_data=print_data)




                



    matrix_sum = np.round(np.sum(node_dist),2)
    assert matrix_sum == 1, \
        'matrix must sum to 1, but is {}'.format(matrix_sum)
    
    if print_data:
        print('Normalised matrix:\n{}'.format(node_dist))
        print('Normalised matrix sum: {}'.format(matrix_sum))
    if path_to_save is not None:
        tools.pickle_data(path, node_dist)
    if plot_fig or show_fig:
        fig = plot_dists.plot_node_dist(node_dist=node_dist, 
                                        eps=eps,
                                        show_fig=show_fig)
        return node_dist, fig

    else:
        return node_dist
   

def gen_node_demands(eps,
                     node_dist, 
                     num_demands, 
                     rack_prob_config=None,
                     duplicate=False,
                     path_to_save=None):
    '''Uses node distribution to generate src-dst node pair demands.

    Args:
        eps (list): List of network node endpoints that can act as sources
            & destinations.
        node_dist (numpy array): 2D matrix array of source-destination pair
            probabilities of being chosen.
        num_demands (int): Number of src-dst node pairs to generate.
        duplicate (bool): Whether or not to duplicate src-dst node pairs. Use
            this is demands you're generating have a 'take down' event as well
            as an 'establish' event.
        path_to_save (str): Path to directory (with file name included) in which
            to save generated distribution. E.g. path_to_save='data/dists/my_dist'.

    Returns:
        tuple: Tuple containing:
            - **sn** (*numpy array*): Selected source nodes.
            - **dn** (*numpy array*): Selected destination nodes.

    '''
    matrix_sum = np.round(np.sum(node_dist),2)
    assert matrix_sum == 1, \
        'demand dist matrix must sum to 1, but is {}'.format(matrix_sum)
    
    # init
    if duplicate:
        sn = np.array(np.zeros((2*num_demands)),dtype=object)
        dn = np.array(np.zeros((2*num_demands)), dtype=object)
    else:
        sn = np.array(np.zeros((num_demands)),dtype=object)
        dn = np.array(np.zeros((num_demands)), dtype=object)

    # source nodes
    sn[:num_demands] = gen_demand_nodes(eps=eps,
                                        node_dist=node_dist,
                                        size=num_demands, 
                                        axis=0)
    if duplicate:
        sn[num_demands:] = sn[:num_demands]

    # destination nodes
    dn[:num_demands] = gen_demand_nodes(eps=eps,
                                        node_dist=node_dist, 
                                        size=num_demands, 
                                        axis=1)
    if duplicate:
        dn[num_demands:] = dn[:num_demands]

    # remove any src-dst conflicts
    for request in np.arange(num_demands):
        while sn[request] == dn[request]:
            dn[request] = gen_demand_nodes(eps=eps,
                                           node_dist=node_dist, 
                                           size=1, 
                                           axis=1)[0]
    if duplicate:
        dn[num_demands:] = dn[:num_demands] # duplicate

    if rack_prob_config is not None:
        # adjust for rack config
        # switch racks_dict keys and values to make hasing easier
        racks_dict = {}
        for key, val in rack_prob_config['racks_dict'].items():
            for v in val:
                if v not in racks_dict.keys():
                    racks_dict[v] = key

        for request in np.arange(num_demands):
            # sample if connection should be intra or inter rack
            inter_rack = np.random.choice(a=[True, False], 
                                          p=[rack_prob_config['prob_inter_rack'], 
                                             1-rack_prob_config['prob_inter_rack']])

            # sample destination node given inter_rack config
            dn[request] = get_suitable_destination_node_for_rack_config(sn[request], 
                                                                        node_dist, 
                                                                        eps, 
                                                                        ep_to_rack=racks_dict, 
                                                                        rack_to_ep=rack_prob_config['racks_dict'], 
                                                                        inter_rack=inter_rack)

            # start = time.time()
            # if inter_rack:
                # # src and dst must be in different racks
                # while racks_dict[sn[request]] == racks_dict[dn[request]]:
                    # dn[request] = gen_demand_nodes(eps=eps,
                                                   # node_dist=node_dist, 
                                                   # size=1, 
                                                   # axis=1)[0]
                    # if time.time()-start > 20:
                        # raise Exception('Cannot find src and dst in different racks. Consider adding more racks, reducing number of endpoints per rack, or changing rack_prob_config.')
            # else:
                # # src and dst must be in same rack
                # while racks_dict[sn[request]] != racks_dict[dn[request]]:
                    # dn[request] = gen_demand_nodes(eps=eps,
                                                   # node_dist=node_dist, 
                                                   # size=1, 
                                                   # axis=1)[0]
                    # if time.time()-start > 20:
                        # raise Exception('Cannot find src and dst in same rack. Consider removing racks, increasing number of endpoints per rack, or changing rack_prob_config.')

    if duplicate:
        dn[num_demands:] = dn[:num_demands]

    if path_to_save is not None:
        data = {'sn': sn, 'dn': dn}
        tools.pickle_data(path_to_save, data)

    return sn, dn


def gen_demand_nodes(eps,
                     node_dist, 
                     size, 
                     axis,
                     path_to_save=None):
    '''Generates demand nodes following the node_dist distribution

    Args:
        eps (list): List of node endpoint labels.
        node_dist (numpy array): Probability distribution each node is chosen
        size (int): Number of demand nodes to generate
        axis (int, 1 or 0): Which axis of normalised node distribution to consider.
            E.g. If generating src nodes, axis=0. If dst nodes, axis=1
        path_to_save (str): Path to directory (with file name included) in which
            to save generated distribution. E.g. path_to_save='data/dists/my_dist'.
    '''
    matrix_sum = np.round(np.sum(node_dist),2)
    assert matrix_sum == 1, \
        'matrix must sum to 1, but is {}'.format(matrix_sum)


    if len(eps) != len(np.sum(node_dist,axis=axis)):
        # must index node dist
        num_nodes, num_pairs, node_to_index, index_to_node = tools.get_network_params(eps)
        ep_node_indices = [node_to_index[ep] for ep in eps]
        node_dist = node_dist[ep_node_indices]

    # make sure sums to 1 (sometimes python floating point arithmetic causes incorrect rounding, and will adjust if used above node indexing)
    probs = adjust_probability_array_sum(np.sum(node_dist, axis=axis), target_sum=1, print_data=False)

    nodes = np.random.choice(a = eps, 
                             size = size,
                             p = probs).astype(object)
    
    if path_to_save is not None:
        tools.pickle_data(path_to_save, nodes)
    
    return nodes



def adjust_probability_array_sum(probs, target_sum=1, print_data=False):
    # check probabilites sum to 1 (python floating point arithmetic can cause problems so need this)
    adjusted_probs = copy.deepcopy(probs)

    total = np.sum(adjusted_probs)
    if print_data:
        print('Initial sum: {}'.format(total))
    difference = total - target_sum
    if difference != 0:
        # need slight adjustment so probs sum to target
        diff_per_pair = difference/len(adjusted_probs)
        for idx in range(len(adjusted_probs)):
            adjusted_probs[idx] -= diff_per_pair
    if print_data:
        print('Final sum: {}'.format(np.sum(adjusted_probs)))

    return adjusted_probs





