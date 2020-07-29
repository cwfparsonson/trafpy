from trafpy.src.dists import plot_dists, val_dists 

import sys
import numpy as np


def get_network_params(eps):
    '''
    Returns basic params of network
    '''
    num_nodes = len(eps)
    num_pairs = np.int(((num_nodes**2) - num_nodes)/2)
    node_indices = [index for index in range(num_nodes)]
    iterables = zip(eps, node_indices)
    node_to_index = {node: index for node, index in iterables}
    iterables = zip(node_indices, eps)
    index_to_node = {index: node for index, node in iterables}
    
    return num_nodes, num_pairs, node_to_index, index_to_node


def assign_probs_to_matrix(eps, probs, matrix):
    num_nodes, num_pairs, node_to_index, index_to_node = get_network_params(eps)
    
    iter = np.nditer(probs)
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
                          path_to_save=None, 
                          plot_fig=False, 
                          show_fig=False,
                          print_data=False):
    '''
    Generates a uniform node distribution where each node has equal
    probability of being chosen
    
    Args:
    - eps (list): List of node endpoints that can act as sources
    & destinations
    
    Returns:
    - node_dist (array of floats): array of src-dst pairs and 
    their probabilities of being chosen during routing session
    - (optional) fig: plotted figure
    '''
    # init network params
    num_nodes, num_pairs, node_to_index, index_to_node = get_network_params(eps)
    node_dist = np.zeros((num_nodes, num_nodes))
    
    # uniform prob each pair chosen
    prob_pair_chosen = (np.ones((num_pairs))/((num_pairs)))/2

    # assign probabilites to matrix
    node_dist = assign_probs_to_matrix(eps=eps,
                                       probs=prob_pair_chosen,
                                       matrix=node_dist)

    matrix_sum = np.round(np.sum(node_dist),2)

    assert matrix_sum == 1, \
        'matrix must sum to 1, but is {}'.format(matrix_sum)
    
    if print_data:
        print('Normalised matrix:\n{}'.format(node_dist))
        print('Normalised matrix sum: {}'.format(matrix_sum))
    if path_to_save is not None:
        val_dists.save_data(path, node_dist)
    if plot_fig or show_fig:
        fig = plot_dists.plot_node_dist(node_dist=node_dist, 
                                        node_to_index_dict=node_to_index, 
                                        eps=eps,
                                        show_fig=show_fig)
        return node_dist, fig

    else:
        return node_dist





def gen_uniform_multinomial_exp_node_dist(eps,
                                          path_to_save=None,
                                          plot_fig=False,
                                          show_fig=False,
                                          print_data=False):
    '''
    Runs a multinomial experiment where each node pair has same (uniform)
    probability of being chosen. Will generate a node demand distribution
    where a few pairs & nodes have a slight skew in demand
    
    Args:
    - eps (list): List of node endpoints that can act as sources
    & destinations
    
    Returns:
    - node_dist (array): array of src-dst pairs and their probabilities 
    of being chosen during routing session
    - (optional) fig: plotted figure
    '''
    # initialise graph params
    num_nodes, num_pairs, node_to_index, index_to_node = get_network_params(eps)
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

    matrix_sum = np.round(np.sum(node_dist),2)

    assert matrix_sum == 1, \
        'matrix must sum to 1, but is {}'.format(matrix_sum)
    
    if print_data:
        print('Normalised matrix:\n{}'.format(node_dist))
        print('Normalised matrix sum: {}'.format(matrix_sum))
    if path_to_save is not None:
        val_dists.save_data(path, node_dist)
    if plot_fig or show_fig:
        fig = plot_dists.plot_node_dist(node_dist=node_dist, 
                                        node_to_index_dict=node_to_index, 
                                        eps=eps,
                                        show_fig=show_fig)
        return node_dist, fig

    else:
        return node_dist


def gen_multimodal_node_dist(eps,
                             skewed_nodes=[],
                             skewed_node_probs=[],
                             num_skewed_nodes=None,
                             path_to_save=None,
                             plot_fig=False,
                             show_fig=False,
                             print_data=False):
    '''
    Generates a multimodal node demand distribution i.e. certain nodes
    have a certain specified probability of being chosen. If no
    skewed nodes given, randomly selects random no. node(s) to skew. If no 
    skew node probabilities given, random selects probability with which
    to skew the node between 0.5 and 0.8. If no num skewed nodes given,
    randomly chooses number of nodes to skew
    
    Args:
    - eps (list): List of node endpoints that can act as sources
    & destinations
    - skewed_nodes (list of ints): Node(s) to whose probability of being
    chosen you want to skew/specify
    - skewed_node_probs (list of floats): Probabilit(y)(ies) of node(s)
    being chosen/specified skews
    - num_skewed_nodes (int): Number of nodes to skew. If None, will gen
    a number between 10% and 30% of the total number of nodes in network
    
    Returns:
    - node_dist (array): array of src-dst pairs and 
    their probabilities of being chosen during routing session
    - (optional) fig: plotted figure
    ''' 
    # initialise graph params
    num_nodes, num_pairs, node_to_index, index_to_node = get_network_params(eps)
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
        print('Prob pair chosen: {}'.format(prob_pair_chosen))

    # assign probabilites to normalised demand matrix
    node_dist = assign_probs_to_matrix(eps=eps,
                                       probs=prob_pair_chosen,
                                       matrix=node_dist)

    matrix_sum = np.round(np.sum(node_dist),2)
    assert matrix_sum == 1, \
        'matrix must sum to 1, but is {}'.format(matrix_sum)
    
    if print_data:
        print('Normalised matrix:\n{}'.format(node_dist))
        print('Normalised matrix sum: {}'.format(matrix_sum))
    if path_to_save is not None:
        val_dists.save_data(path, node_dist)
    if plot_fig or show_fig:
        fig = plot_dists.plot_node_dist(node_dist=node_dist, 
                                        node_to_index_dict=node_to_index, 
                                        eps=eps,
                                        show_fig=show_fig)
        return node_dist, fig

    else:
        return node_dist


def gen_multimodal_node_pair_dist(eps,
                                  skewed_pairs = [],
                                  skewed_pair_probs = [],
                                  num_skewed_pairs=None,
                                  path_to_save=None,
                                  plot_fig=False,
                                  show_fig=False,
                                  print_data=False):
    ''' 
    Generates a multimodal node pair demand distribution i.e. certain node
    pairs have a certain specified probability of being chosen. If no
    skewed pairs given, randomly selects pair to skew. If no skew 
    pair probabilities given, random selects probability with which
    to skew the pair between 0.1 and 0.3. If no num skewed pairs given,
    randomly chooses number of pairs to skew

    Args:
    - eps (list): List of node endpoints that can act as sources
    & destinations
    - skewed_pairs (list of lists): List of the node pairs [src,dst] to 
    skew
    - skewed_pair_probs (list of floats): Probabilities of node pairs being
    chosen
    - num_skewed_pairs (int): Number of pairs to randomly skew
    
    Returns:
    - node_dist (array): array of src-dst pairs and 
    their probabilities of being chosen during routing session
    - (optional) fig: plotted figure
    '''
    # initialise graph params
    num_nodes, num_pairs, node_to_index, index_to_node = get_network_params(eps)
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

    matrix_sum = np.round(np.sum(node_dist),2)
    assert matrix_sum == 1, \
        'matrix must sum to 1, but is {}'.format(matrix_sum)
    
    if print_data:
        print('Normalised matrix:\n{}'.format(node_dist))
        print('Normalised matrix sum: {}'.format(matrix_sum))
    if path_to_save is not None:
        val_dists.save_data(path, node_dist)
    if plot_fig or show_fig:
        fig = plot_dists.plot_node_dist(node_dist=node_dist, 
                                        node_to_index_dict=node_to_index, 
                                        eps=eps,
                                        show_fig=show_fig)
        return node_dist, fig

    else:
        return node_dist
    

def gen_demand_nodes(eps,
                     node_dist, 
                     size, 
                     axis,
                     path_to_save=None):
    '''
    Generates demand nodes following the node_dist distribution

    Args:
    - eps (list): list of node endpoint labels
    - node_dist (array): probability distribution each node is chosen
    - size (int): number of demand nodes to generate
    - axis (binary): which axis of normalised node distribution to consider.
    E.g. If generating src nodes, axis=0. If dst nodes, axis=1
    '''
    matrix_sum = np.round(np.sum(node_dist),2)
    assert matrix_sum == 1, \
        'matrix must sum to 1, but is {}'.format(matrix_sum)

    nodes = np.random.choice(a = eps, 
                             size = size,
                             p = np.sum(node_dist,
                                        axis=axis)).astype(object)
    
    if path_to_save is not None:
        save_data(path_to_save, nodes)
    
    return nodes



