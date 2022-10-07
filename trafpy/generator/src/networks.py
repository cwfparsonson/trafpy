'''Module for generating and plotting networks.'''

from trafpy.generator.src import tools
import copy
import networkx as nx
import matplotlib.pyplot as plt
import json


def gen_arbitrary_network(num_eps,
                          ep_label=None, 
                          ep_capacity=12500, 
                          num_channels=1, 
                          racks_dict=None,
                          topology_type=None,
                          **kwargs):
    '''Generates an arbitrary network with num_eps nodes labelled as ep_label.

    Note that no edges are formed in this network; it is purely for ep name 
    indexing purposes when using Demand class. This is useful where want to
    use the demand class but not necessarily with a carefully crafted networkx
    graph that accurately mimics the network you will use for the demands

    Args:
        num_eps (int): Number of endpoints in network.
        ep_label (str,int,float): Endpoint label (e.g. 'server'). All endpoints will have
            ep_label appended to the start of their label (e.g. 'server_0', 'server_1', ...).
        ep_capacity (int, float): Byte capacity per end point channel.
        num_channels (int, float): Number of channels on each link in network.
        racks_dict (dict): Mapping of which end points are in which racks. Keys are
            rack ids, values are list of end points. If None, assume there is not
            clustering/rack system in the network where have different end points
            in different clusters/racks.

    Returns:
        networkx graph: network object

    '''
    network = nx.Graph()
    network.add_nodes_from([node for node in range(num_eps)])
    
    if ep_label is None:
        # must be str or not json serialisable
        servers = [str(i) for i in range(num_eps)]
    else:
        servers = [ep_label+'_'+str(i) for i in range(num_eps)]
    relabel_mapping = {node: label for node, label in zip(range(num_eps),servers)}
    network = nx.relabel_nodes(network, relabel_mapping)
    eps = []
    for node in list(network.nodes):
        try:
            if ep_label in node:
                eps.append(node)
        except TypeError:
            # ep_label is None
            eps.append(node)
    network.graph['endpoints'] = eps

    # /= 2 to get max theoretical capacity (number of units which network can transfer per unit time)
    max_nw_capacity = (num_eps * ep_capacity * num_channels) / 2

    if topology_type is None:
        topology_type = 'arbitrary_endpoints_{}_chancap_{}_channels_{}'.format(num_eps, ep_capacity, num_channels)

    init_global_network_attrs(network,
                              max_nw_capacity,
                              num_channels,
                              ep_link_capacity=ep_capacity*num_channels,
                              endpoint_label=ep_label,
                              node_labels=[ep_label],
                              racks_dict=racks_dict,
                              topology_type=topology_type)
    
    return network



def gen_nsfnet_network(ep_label='server', 
                       rack_label='rack',
                       N=0, 
                       num_channels=2, 
                       server_to_rack_channel_capacity=1,
                       rack_to_rack_channel_capacity=10,
                       show_fig=False):
    '''Generates the standard 14-node NSFNET topology (a U.S. core network).
    
    Args:
        ep_label (str,int,float): Endpoint label (e.g. 'server'). All endpoints will have
            ep_label appended to the start of their label (e.g. 'server_0', 'server_1', ...).
        N (int): Number of servers per rack. If 0, assume all nodes in nsfnet
            are endpoints
        num_channels (int,float): Number of channels on each link in network.
        server_to_rack_channel_capacity (int,float): Byte capacity per channel 
            between servers and ToR switch.
        rack_to_rack_channel_capacity (int,float): Byte capacity per channel between racks.
        show_fig (bool): Whether or not to plot and show fig. If True, will
            display fig.

    Returns:
        networkx graph: network object

    '''
    channel_names = gen_channel_names(num_channels)
    network = nx.Graph()

    node_pair_list = [[0,1],
                      [0,3],
                      [0,2],
                      [1,2],
                      [1,7],
                      [3,8],
                      [3,4],
                      [3,6],
                      [4,5],
                      [4,5],
                      [5,2],
                      [5,13],
                      [5,12],
                      [6,7],
                      [7,10],
                      [8,11],
                      [8,9],
                      [9,10],
                      [9,12],
                      [10,11],
                      [10,13],
                      [11,12]]

    if N == 0:
        # above nodes are all end points
        label = ep_label
    else:
        # above nodes are ToR switch nodes
        label = rack_label
    for idx in range(len(node_pair_list)):
        node_pair_list[idx][0] = label + '_' + str(node_pair_list[idx][0])
        node_pair_list[idx][1] = label + '_' + str(node_pair_list[idx][1])

    # add 14 nodes
    for edge in node_pair_list:
        network.add_edge(*tuple(edge))

    if N == 0:
        # assume all nodes are servers
        racks_dict = None
    else:
        # each of 14 nodes in NSFNET is a ToR switch
        i = 0
        racks_dict = {rack: [] for rack in range(14)}
        for rack in range(14):
            for server in range(N):
                racks_dict[rack].append(ep_label+'_'+str(i))
                network.add_edge(ep_label+'_'+str(i), rack_label+'_'+str(rack))
                i += 1
    
    channel_names = gen_channel_names(num_channels)
    edges = [edge for edge in network.edges]
    add_edges_capacity_attrs(network, edges, channel_names, rack_to_rack_channel_capacity)

    # set gloabl network attrs
    network.graph['endpoints'] = get_endpoints(network, ep_label)

    # /= 2 to get max theoretical capacity (number of units which network can transfer per unit time)
    max_nw_capacity = (len(network.edges) * num_channels * rack_to_rack_channel_capacity) / 2

    init_global_network_attrs(network, 
                            max_nw_capacity, 
                            num_channels, 
                            ep_link_capacity=server_to_rack_channel_capacity*num_channels,
                            endpoint_label=ep_label,
                            node_labels=[ep_label, rack_label],
                            topology_type='14_node_nsfnet',
                            racks_dict=racks_dict)
    if show_fig:
        plot_network(network, show_fig=True)
    
    return network

def gen_simple_network(ep_label='server', 
                       num_channels=2, 
                       server_to_rack_channel_capacity=500,
                       show_fig=False):
    '''Generates very simple 5-node topology.

    Args:
        ep_label (str,int,float): Endpoint label (e.g. 'server'). All endpoints will have
            ep_label appended to the start of their label (e.g. 'server_0', 'server_1', ...).
        num_channels (int,float): Number of channels on each link in network.
        channel_capacity (int,float): Byte capacity per channel.
        show_fig (bool): Whether or not to plot and show fig. If True, will
            display fig.

    Returns:
        networkx graph: network object

    '''
    network = nx.Graph()
    network.add_nodes_from([node for node in range(5)])
    network.add_edges_from([(0,1),
                          (0,2),
                          (1,2),
                          (2,4),
                          (4,3),
                          (3,1)],weight=1)
    servers = [ep_label+'_'+str(i) for i in range(5)]
    relabel_mapping = {node: label for node, label in zip(range(5),servers)}
    network = nx.relabel_nodes(network, relabel_mapping)

    channel_names = gen_channel_names(num_channels)
    edges = [edge for edge in network.edges]
    add_edges_capacity_attrs(network, edges, channel_names, server_to_rack_channel_capacity)

    # set gloabl network attrs
    network.graph['endpoints'] = get_endpoints(network, ep_label)

    # /= 2 to get max theoretical capacity (number of units which network can transfer per unit time)
    max_nw_capacity = (len(network.edges) * num_channels * server_to_rack_channel_capacity) / 2

    init_global_network_attrs(network, 
                            max_nw_capacity, 
                            num_channels, 
                            ep_link_capacity=server_to_rack_channel_capacity*num_channels,
                            endpoint_label=ep_label,
                            node_labels=[ep_label],
                            topology_type='5_node_simple_network')

    if show_fig:
        plot_network(network, show_fig=True)

    
    return network

def get_endpoints(network, ep_label):
    '''Gets list of endpoints of network.

    Args:
        network (networkx graph): Networkx object.
        ep_label (str,int,float): Endpoint label (e.g. 'server'). All endpoints will have
            ep_label appended to the start of their label (e.g. 'server_0', 'server_1', ...).

    Returns:
        eps (list): List of endpoints.

    '''
    eps = []
    for node in list(network.nodes):
        if ep_label in node:
            eps.append(node)

    return eps

def gen_fat_tree(k=4,
                  L=2,
                  n=4,
                  ep_label='server',
                  rack_label='rack',
                  edge_label='edge',
                  aggregate_label='agg',
                  core_label='core',
                  num_channels = 2,
                  server_to_rack_channel_capacity=500,
                  rack_to_edge_channel_capacity=1000,
                  edge_to_agg_channel_capacity=1000,
                  agg_to_core_channel_capacity=2000,
                  rack_to_core_channel_capacity=2000,
                  show_fig=False):
    '''Generates a perfect fat tree (i.e. all layers have switches with same radix/number of ports).

    Top layer is always core (spine) switch layer, bottom layer is always
    ToR (leaf) layer.

    L must be either 2 (core, ToR) or 4 (core, agg, edge, ToR)

    N.B. L=2 is commonly referred to as '2-layer Clos' or 'Clos' or 'spine-leaf' topology

    Resource for building (scroll down to summary table with equations):

    https://packetpushers.net/demystifying-dcn-topologies-clos-fat-trees-part2/

    Another good resource for data centre topologies etc. in general:

    https://www.oreilly.com/library/view/bgp-in-the/9781491983416/ch01.html#:~:text=The%20most%20common%20routing%20protocol,single%20data%20center%2C%20as%20well.

    Parameters of network:

    - number of core (spine) switches = (k/2)^(L/2) (top layer)
    - number of edge switches (if L=4) = (k^2)/2
    - number of agg switches (if L=4) = (k^2)/2
    - number of pods (if L=4) (pod is a group of agg and/or edge switches) = 2*(k/2)^(L-2)
    - number of ToR (leaf) switches (racks) = 2*(k/2)^(L-1) (bottom layer)
    - number of server-facing ToR 'host' ports = 2*(k/2)^2 (can have multiple servers connected to same host port, & can oversubscribe)
    - number of servers = number ToR switches * n

    Args:
        k (int): Number of ports (links) on each switch (both up and down).
        L (int): Number of layers in the fat tree.
        n (int): Number of server per rack.
        ep_label (str,int,float): Endpoint label (e.g. 'server'). All endpoints will have
            ep_label appended to the start of their label (e.g. 'server_0', 'server_1', ...).
        edge_label (str,int): Label to assign to edge switch nodes
        aggregate_label (str,int): Label to assign to edge switch nodes
        core_label (str,int): Label to assign to core switch nodes
        num_channels (int, float): Number of channels on each link in network
        server_to_edge_channel_capacity (int,float): Byte capacity per channel
        edge_to_agg_channel_capacity (int,float): (if L==4) Byte capacity per channel
        agg_to_core_channel_capacity (int,float): (if L==4) Byte capacity per channel
        rack_to_core_channel_capacity (int,float): (if L==2) Byte capacity per channel

    Returns:
        networkx graph: network object

    '''
    if L != 2 and L != 4:
        raise Exception('L must be 2 (ToR layer, core layer) or 4 (ToR layer, edge layer, agg layer, core layer), but is {}.'.format(L))
    if k % 2 != 0:
        raise Exception('k must be even since, in perfect fat tree, have equal number of up and down ports on each switch, but is {}.'.format(k))

    channel_names = gen_channel_names(num_channels)

    # initialise network nodes
    if L == 2:
        node_labels = [ep_label, rack_label, core_label]
    else:
        node_labels = [ep_label, rack_label, edge_label, aggregate_label, core_label]

    #num_cores = int((k/2)**(L-1))
    #num_cores = int((k/2)**2)
    num_cores = int((k/2)**(L/2))
    num_aggs = int((k**2)/2)
    num_edges = int((k**2)/2)
    num_pods = int(2*(k/2)**(L-2))
    num_racks = int(2*(k/2)**(L-1))
    num_servers = int(num_racks * n)

    cores = [core_label+'_'+str(i) for i in range(num_cores)]
    aggs = [aggregate_label+'_'+str(i) for i in range(num_aggs)]
    edges = [edge_label+'_'+str(i) for i in range(num_edges)]
    racks = [rack_label+'_'+str(i) for i in range(num_racks)]
    servers = [ep_label+'_'+str(i) for i in range(num_servers)]

    # create core and rack layer networks
    core_layer = nx.Graph()
    rack_layer = nx.Graph()
    core_layer.add_nodes_from(cores)
    rack_layer.add_nodes_from(racks)

    # combine cores and racks into single network
    fat_tree_network = nx.compose(core_layer, rack_layer)
    
    if L == 2:
        # 2 layers: Core, ToR
        # link racks to cores, add link attributes
        rack_iterator = iter(racks)
        for rack in racks:
            core_iterator = iter(cores)
            # have k/2 up-ports on each switch
            for up_port in range(int(k/2)):
                core = next(core_iterator)
                fat_tree_network.add_edge(rack, core)
                add_edge_capacity_attrs(fat_tree_network,
                                        (rack, core),
                                        channel_names,
                                        rack_to_core_channel_capacity)
    else:
        # 4 layers: Core, Agg, Edge, ToR. Agg and Edge switches grouped into pods.
        # group edges and aggregates into pods
        num_pods = int(k)
        pods = [[] for i in range(num_pods)]
        prev_iter = 0
        for pod_iter in range(len(pods)):
            curr_iter = int(prev_iter + (k/2))
            pods[pod_iter].append(edges[prev_iter:curr_iter])
            pods[pod_iter].append(aggs[prev_iter:curr_iter])
            prev_iter = curr_iter

        # create dict of pod networks
        pod_labels = ['pod_'+str(i) for i in range(num_pods)]
        pods_dict = {tuple([pod]): nx.Graph() for pod in pod_labels}
        for pod_iter in range(num_pods):
            key = ('pod_'+str(pod_iter),)
            pod_edges = pods[pod_iter][0]

            pod_aggs = pods[pod_iter][1]
            pods_dict[key].add_nodes_from(pod_edges)
            pods_dict[key].add_nodes_from(pod_aggs)

            # connect edge and aggregate switches within pod, add link attributes
            for pod_edge in pod_edges:
                for pod_agg in pod_aggs:
                    pods_dict[key].add_edge(pod_agg, pod_edge)
                    add_edge_capacity_attrs(pods_dict[key], 
                                     (pod_agg,pod_edge), 
                                     channel_names, 
                                     edge_to_agg_channel_capacity)

        # add pods (agg + edge) layer to fat-tree
        pod_networks = list(pods_dict.values())
        for pod_iter in range(num_pods):
            fat_tree_network = nx.compose(fat_tree_network, pod_networks[pod_iter])

        # link aggregate switches in pods to core switches, add link attributes
        for pod_iter in range(num_pods):
            pod_aggs = pods[pod_iter][1]
            core_iterator = iter(cores)
            for pod_agg in pod_aggs:
                while fat_tree_network.degree[pod_agg] < k:
                    core = next(core_iterator)
                    fat_tree_network.add_edge(core, pod_agg)
                    add_edge_capacity_attrs(fat_tree_network,
                                     (core,pod_agg),
                                     channel_names,
                                     agg_to_core_channel_capacity)

        # link edge switches in pods to racks, add link attributes
        rack_iterator = iter(racks)
        for pod_iter in range(num_pods):
            pod_edges = pods[pod_iter][0]
            for pod_edge in pod_edges:
                while fat_tree_network.degree[pod_edge] < k:
                    rack = next(rack_iterator)
                    fat_tree_network.add_edge(pod_edge, rack)
                    add_edge_capacity_attrs(fat_tree_network,
                                     (pod_edge,rack),
                                     channel_names,
                                     rack_to_edge_channel_capacity)

    # link servers to racks, add link attributes
    racks_dict = {rack: [] for rack in racks} # track which endpoints in which rack
    server_iterator = iter(servers)
    for rack in racks:
        for _ in range(n):
            server = next(server_iterator)
            fat_tree_network.add_edge(rack, server)
            add_edge_capacity_attrs(fat_tree_network,
                                    (rack, server),
                                    channel_names,
                                    server_to_rack_channel_capacity)
            racks_dict[rack].append(server)

    # calc total network capacity
    # /= 2 to get max theoretical capacity (number of units which network can transfer per unit time)
    max_nw_capacity = (num_servers * num_channels * server_to_rack_channel_capacity) / 2


    # init global network attrs
    fat_tree_network.graph['endpoints'] = servers
    init_global_network_attrs(fat_tree_network, 
                              max_nw_capacity, 
                              num_channels, 
                              ep_link_capacity=server_to_rack_channel_capacity*num_channels,
                              endpoint_label=ep_label,
                              node_labels=node_labels,
                              topology_type='fat_tree',
                              racks_dict=racks_dict)

    if show_fig:
        plot_network(fat_tree_network, show_fig=True)

    return fat_tree_network

    


def init_global_network_attrs(network, 
                              max_nw_capacity, 
                              num_channels, 
                              ep_link_capacity,
                              endpoint_label = 'server',
                              topology_type='unknown', 
                              node_labels=['server'],
                              racks_dict=None):
    '''Initialises the standard global network attributes of a given network.

    Args:
        network (obj): NetworkX object.
        max_nw_capacity (int/float): Maximum rate at which info can be reliably 
            transmitted over the network (sum of all link capacities).
        num_channels (int): Number of channels on each link in network.
        topology_type (str): Label of network topology (e.g. 'fat_tree').
        node_labels (list): Label classes assigned to network nodes 
            (e.g. ['server', 'rack', 'edge']).
        racks_dict (dict): Which servers/endpoints are in which rack. If None,
            assume do not have rack system where have multiple servers in one
            rack.

    '''
    network.graph['endpoint_label'] = endpoint_label
    network.graph['num_channels_per_link'] = num_channels
    network.graph['ep_link_capacity'] = ep_link_capacity
    network.graph['ep_link_port_capacity'] = ep_link_capacity / 2 # all eps have a src & a dst port
    network.graph['max_nw_capacity'] = max_nw_capacity
    network.graph['curr_nw_capacity_used'] = 0
    network.graph['num_active_connections'] = 0
    network.graph['total_connections_blocked'] = 0
    network.graph['node_labels'] = node_labels
    network.graph['topology_type'] = topology_type
    network.graph['channel_names'] = gen_channel_names(num_channels)

    # ensure racks dict is str so json serialisable
    if racks_dict is not None:
        _racks_dict = {}
        for key, val in racks_dict.items():
            _racks_dict[str(key)] = []
            for v in val:
                _racks_dict[str(key)].append(str(v))
        network.graph['rack_to_ep_dict'] = _racks_dict
    else:
        network.graph['rack_to_ep_dict'] = None

    if racks_dict is not None:
        # switch racks_dict keys and values to make hashing easier
        ep_to_rack_dict = {}
        for key, val in _racks_dict.items():
            for v in val:
                if v not in ep_to_rack_dict.keys():
                    ep_to_rack_dict[v] = key
        network.graph['ep_to_rack_dict'] = ep_to_rack_dict
    else:
        network.graph['ep_to_rack_dict'] = None


def gen_channel_names(num_channels):
    '''Generates channel names for channels on each link in network.'''
    channels = [channel+1 for channel in range(num_channels)]
    channel_names = ['channel_' + str(channel) for channel in channels]
    
    return channel_names

def add_edge_capacity_attrs(network, 
                            edge, 
                            channel_names, 
                            channel_capacity, 
                            bidirectional_links=True):
    '''Adds channels and corresponding max channel bytes to single edge in network.
    
    Args:
        network (networkx graph): Network containing edges to whiich attrs will
            be added.
        edge (tuple): Node-node edge pair.
        channel_names (list): List of channel names to add to edge.
        channel_capacity (int,float): Capacity to allocate to each channel.
        bidirectional_links (bool): If True, each link has capacity split equally
            between src and dst port. I.e. all links have a src and dst port
            which are treated separately to incoming and outgoing traffic to and
            from given node (switch or server).

    '''
    if bidirectional_links:
        attrs = {edge:
                    {'{}_to_{}_port'.format(edge[0], edge[1]):
                        {'channels':
                            {channel: channel_capacity/2 for channel in channel_names},
                         'max_channel_capacity': channel_capacity/2
                         },
                     '{}_to_{}_port'.format(edge[1], edge[0]):
                        {'channels':
                            {channel: channel_capacity/2 for channel in channel_names},
                         'max_channel_capacity': channel_capacity/2
                         }
                        }
                    }
                    
    else:
        attrs = {edge:
                    {'channels': {channel: channel_capacity for channel in channel_names},
                     'max_channel_capacity': channel_capacity}}
    
    nx.set_edge_attributes(network, attrs)




def add_edges_capacity_attrs(network, 
                             edges,
                             channel_names,
                             channel_capacity,
                             bidirectional_links=True):
    '''Adds channels & max channel capacitys to single edge in network.
    
    To access e.g. the edge going from node 0 to node 1 (edge (0, 1)), you
    would index the network with network[0][1]

    To access e.g. the channel_1 attribute of this particular (0, 1) edge, you
    would do network[0][1]['channels']['channel_1']
    OR
    if bidirectional_links, you do network[0][1]['0_to_1_port']['channels']['channel_1']
    or network[0][1]['1_to_0_port']['channels']['channel_1] depending on which direction
    of the link you want to access.
    
    Args:
        network (networkx graph): Network containing edges to which attrs will
            be added.
        edges (list): List of node pairs in tuples.
        channel_names (list of str): List of channel names to add to edge.
        channel_capacity (int, float): Capacity to allocate to each channel.
        bidirectional_links (bool): If True, each link has capacity split equally
            between src and dst port. I.e. all links have a src and dst port
            which are treated separately to incoming and outgoing traffic to and
            from given node (switch or server).

    '''
    if bidirectional_links:
        attrs = {edge:
                    {'{}_to_{}_port'.format(edge[0], edge[1]):
                        {'channels':
                            {channel: channel_capacity/2 for channel in channel_names},
                         'max_channel_capacity': channel_capacity/2
                         },
                     '{}_to_{}_port'.format(edge[1], edge[0]):
                        {'channels':
                            {channel: channel_capacity/2 for channel in channel_names},
                         'max_channel_capacity': channel_capacity/2
                         }
                        }
                    for edge in edges}
    else:
        attrs = {edge: 
                    {'channels': 
                        {channel: channel_capacity for channel in channel_names},
                     'max_channel_capacity': 
                        channel_capacity
                     } for edge in edges}

    nx.set_edge_attributes(network, attrs)
  

def get_node_type_dict(network, node_types=[]):
    '''Gets dict where keys are node types, values are list of nodes for each node type in graph.'''
    network_nodes = []
    for network_node in network.nodes:
        network_nodes.append(network_node)
    network_nodes_dict = {node_type: [] for node_type in node_types}
    for n in network_nodes:
        for node_type in node_types:
            if node_type in n:
                network_nodes_dict[node_type].append(n)
            else:
                # not this node type
                pass
    
    return network_nodes_dict


def get_fat_tree_positions(net, width_scale=500, height_scale=10):
    '''Gets networkx positions of nodes in fat tree network for plotting.'''
    pos = {}

    node_type_dict = get_node_type_dict(net, net.graph['node_labels'])
    node_types = list(node_type_dict.keys())
    
    heights = {} # dict for heigh separation between fat tree layers
    widths = {} # dict for width separation between nodes within layers
    h = iter([1, 2, 3, 4, 5]) # server, rack, edge, agg, core heights
    for node_type in node_types: 
        heights[node_type] = next(h)
        widths[node_type] = 1/(len(node_type_dict[node_type])+1)
        idx = 0
        for node in node_type_dict[node_type]:
            pos[node] = ((idx+1)*widths[node_type]*width_scale,heights[node_type]*height_scale)
            idx += 1

    return pos
   

def init_network_node_positions(net):
    '''Initialises network node positions for plotting.'''
    if net.graph['topology_type'] == 'fat_tree':
        pos = get_fat_tree_positions(net)

    else:
        pos = nx.nx_agraph.graphviz_layout(net, prog='neato')
    
    return pos


def plot_network(network,
                 draw_node_labels=True,
                 ep_label='server',
                 network_node_size=2000,
                 font_size=30,
                 linewidths=1,
                 fig_scale=2,
                 path_to_save=None, 
                 show_fig=False):
    '''Plots networkx graph.

    Recognises special fat tree network and applies appropriate node positioning,
    labelling, colouring etc.

    Args:
        network (networkx graph): Network object to be plotted.
        draw_node_labels (bool): Whether or not to draw node labels on plot. 
        ep_label (str,int,float): Endpoint label (e.g. 'server'). All endpoints will have
            ep_label appended to the start of their label (e.g. 'server_0', 'server_1', ...).
        network_node_size (int,float): Size of plotted nodes.
        font_size (int,float): Size of of font of plotted labels etc.
        linewidths (int,float): Width of edges in network.
        fig_scale (int,float): Scaling factor to apply to plotted network.
        path_to_save (str): Path to directory (with file name included) in which
            to save generated plot. E.g. path_to_save='data/my_plot'
        show_fig (bool): Whether or not to plot and show fig. If True, will
            return and display fig.
        
    Returns:
        matplotlib.figure.Figure: node distribution plotted as a 2d matrix. 

    '''
    
    net_node_positions = init_network_node_positions(copy.deepcopy(network))

    fig = plt.figure(figsize=[15*fig_scale,15*fig_scale])

    # add nodes and edges
    pos = {}
    network_nodes = []
    network_nodes_dict = get_node_type_dict(network, network.graph['node_labels'])
    for nodes in list(network_nodes_dict.values()):
        for network_node in nodes:
            pos[network_node] = net_node_positions[network_node]
            
    # network nodes
    node_colours = iter(['#25c44d', '#36a0c7', '#e8b017', '#6115a3', '#160e63']) # server, rack, edge, agg, core
    for node_type in network.graph['node_labels']:
        nx.draw_networkx_nodes(network, 
                               pos, 
                               nodelist=network_nodes_dict[node_type],
                               node_size=network_node_size, 
                               node_color=next(node_colours), 
                               linewidths=linewidths, 
                               label=node_type)
    if draw_node_labels:
        # nodes
        nx.draw_networkx_labels(network, 
                                pos, 
                                font_size=font_size, 
                                font_color='k', 
                                font_family='sans-serif', 
                                font_weight='normal', 
                                alpha=1.0)
    
    # fibre links
    fibre_links = list(network.edges)
    nx.draw_networkx_edges(network, 
                           pos,
                           edgelist=fibre_links,
                           edge_color='k',
                           width=3,
                           label='Fibre link')


    if path_to_save is not None:
        tools.pickle_data(path_to_save, fig)

    if show_fig:
        plt.show()

    return fig


if __name__ == '__main__':
    #network = gen_simple_network()
    #network = gen_nsfnet_network()
    network = gen_fat_tree(k=3)
    
    plot_network(network, 'figures/graph/',name='network_graph.png',with_labels=True)
















    



