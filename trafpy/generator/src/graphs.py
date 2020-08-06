import networkx as nx
import matplotlib.pyplot as plt

def gen_nsfnet_graph(ep_label='server', num_channels=2, channel_capacity=10):
    '''
    Generates the standard 14-node NSFNET topology (a U.S. core network)
    
    Args:
    - ep_label (str,int,float): label to apply to end points in graph
    - num_channels (int,float): number of channels on each link in network
    - channel_capacity (int,float): byte capacity per channel

    Returns:
    - Graph (object)
    '''
    channel_names = gen_channel_names(num_channels)
    Graph = nx.Graph()
    Graph.add_nodes_from([node for node in range(14)])
    Graph.add_edges_from([(0,1),
                          (0,3),
                          (0,2),
                          (1,2),
                          (1,7),
                          (3,8),
                          (3,4),
                          (3,6),
                          (4,5),
                          (4,5),
                          (5,2),
                          (5,13),
                          (5,12),
                          (6,7),
                          (7,10),
                          (8,11),
                          (8,9),
                          (9,10),
                          (9,12),
                          (10,11),
                          (10,13),
                          (11,12)], weight=1)
    servers = [ep_label+'_'+str(i) for i in range(14)]
    relabel_mapping = {node: label for node, label in zip(range(14),servers)}
    Graph = nx.relabel_nodes(Graph, relabel_mapping)
    
    channel_names = gen_channel_names(num_channels)
    edges = [edge for edge in Graph.edges]
    add_edges_capacity_attrs(Graph, edges, channel_names, channel_capacity)

    # set gloabl graph attrs
    Graph.graph['endpoints'] = get_endpoints(Graph, ep_label)
    max_nw_capacity = len(Graph.edges) * num_channels * channel_capacity
    init_global_graph_attrs(Graph, 
                            max_nw_capacity, 
                            num_channels, 
                            node_labels=[ep_label],
                            topology_type='14_node_nsfnet')
    
    
    return Graph

def gen_simple_graph(ep_label='server', num_channels=2, channel_capacity=10):
    '''
    Generates very simple 5-node topology

    Args:
    - ep_label (str,int,float): label to apply to end points in graph
    - num_channels (int,float): number of channels on each link in network
    - channel_capacity (int,float): byte capacity per channel

    Returns:
    - Graph (object)
    '''
    Graph = nx.Graph()
    Graph.add_nodes_from([node for node in range(5)])
    Graph.add_edges_from([(0,1),
                          (0,2),
                          (1,2),
                          (2,4),
                          (4,3),
                          (3,1)],weight=1)
    servers = [ep_label+'_'+str(i) for i in range(5)]
    relabel_mapping = {node: label for node, label in zip(range(5),servers)}
    Graph = nx.relabel_nodes(Graph, relabel_mapping)

    channel_names = gen_channel_names(num_channels)
    edges = [edge for edge in Graph.edges]
    add_edges_capacity_attrs(Graph, edges, channel_names, channel_capacity)

    # set gloabl graph attrs
    Graph.graph['endpoints'] = get_endpoints(Graph, ep_label)
    max_nw_capacity = len(Graph.edges) * num_channels * channel_capacity
    init_global_graph_attrs(Graph, 
                            max_nw_capacity, 
                            num_channels, 
                            node_labels=[ep_label],
                            topology_type='5_node_simple_graph')

    
    return Graph

def get_endpoints(graph, ep_label):
    '''
    Gets list of endpoints of graph

    Args:
    - graph (object)
    - ep_label (str): label of end points

    Returns:
    - eps (list): list of eps
    '''
    eps = []
    for node in list(graph.nodes):
        if ep_label in node:
            eps.append(node)

    return eps


def gen_fat_tree(k=4,
                 ep_label='server',
                 edge_label='edge',
                 aggregate_label='agg',
                 core_label='core',
                 num_channels = 2,
                 server_to_edge_channel_capacity=10,
                 edge_to_agg_channel_capacity=40,
                 agg_to_core_channel_capacity=100):
    '''
    Generates a data centre network with a 3-layer fat tree topology
    Resource for building: 
    https://blogchinmaya.blogspot.com/2017/04/what-is-fat-tree
    
    Parameters of network:
    - number of core switches = (k/2)^2
    - number of pods = k
    - number of aggregate switches = (k^2)/2
    - number of edge switches = (k^2)/2
    - number of servers = (k^3)/4

    Args:
    - k (int): number of ports/links on each switch
    - ep_label (str,int): label to assign to end point nodes
    - edge_label (str,int): label to assign to edge switch nodes
    - aggregate_label (str,int): label to assign to edge switch nodes
    - core_label (str,int): label to assign to core switch nodes
    - num_channels (int, float): number of channels on each link in network
    - server_to_edge_channel_capacity (int,float): byte capacity per channel
    - edge_to_agg_channel_capacity (int,float): byte capacity per channel
    - agg_to_core_channel_capacity (int,float): byte capacity per channel

    Returns:
    - fat_tree_graph (object)
    '''
    channel_names = gen_channel_names(num_channels)

    # initialise graph nodes
    num_cores = int((k/2)**2)
    num_aggs = int((k**2)/2)
    num_edges = int((k**2)/2)
    num_servers = int((k**3)/4)

    cores = [core_label+'_'+str(i) for i in range(num_cores)]
    aggs = [aggregate_label+'_'+str(i) for i in range(num_aggs)]
    edges = [edge_label+'_'+str(i) for i in range(num_edges)]
    servers = [ep_label+'_'+str(i) for i in range(num_servers)]
   
    # create core and server layer graphs
    core_layer = nx.Graph()
    server_layer = nx.Graph()
    core_layer.add_nodes_from(cores)
    server_layer.add_nodes_from(servers)
    
    # group edges and aggregates into pods
    num_pods = int(k)
    pods = [[] for i in range(num_pods)]
    prev_iter = 0
    for pod_iter in range(len(pods)):
        curr_iter = int(prev_iter + (k/2))
        pods[pod_iter].append(edges[prev_iter:curr_iter])
        pods[pod_iter].append(aggs[prev_iter:curr_iter])
        prev_iter = curr_iter

    # create dict of pod graphs
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

    # combine cores, pods and servers into single graph
    pod_graphs = list(pods_dict.values())
    fat_tree_graph = nx.compose(core_layer, server_layer)
    for pod_iter in range(num_pods):
        fat_tree_graph = nx.compose(fat_tree_graph, pod_graphs[pod_iter])

    # link aggregate switches in pods to core switches, add link attributes
    for pod_iter in range(num_pods):
        pod_aggs = pods[pod_iter][1]
        core_iterator = iter(cores)
        for pod_agg in pod_aggs:
            while fat_tree_graph.degree[pod_agg] < k:
                core = next(core_iterator)
                fat_tree_graph.add_edge(core, pod_agg)
                add_edge_capacity_attrs(fat_tree_graph,
                                 (core,pod_agg),
                                 channel_names,
                                 agg_to_core_channel_capacity)

    # link edge switches in pods to servers, add link attributes
    server_iterator = iter(servers)
    for pod_iter in range(num_pods):
        pod_edges = pods[pod_iter][0]
        for pod_edge in pod_edges:
            while fat_tree_graph.degree[pod_edge] < k:
                server = next(server_iterator)
                fat_tree_graph.add_edge(pod_edge, server)
                add_edge_capacity_attrs(fat_tree_graph,
                                 (pod_edge,server),
                                 channel_names,
                                 server_to_edge_channel_capacity)

    # add end points as global graph property
    fat_tree_graph.graph['endpoints'] = get_endpoints(fat_tree_graph, ep_label)


    # calc total network byte capacity
    num_agg_core_links = num_cores * k
    num_edge_agg_links = num_pods * k
    num_server_edge_links = num_edges * (k/2)
    agg_core_capacity = num_agg_core_links * agg_to_core_channel_capacity * num_channels
    edge_agg_capacity = num_edge_agg_links * edge_to_agg_channel_capacity * num_channels
    server_edge_capacity = num_server_edge_links * server_to_edge_channel_capacity * num_channels
    max_nw_capacity = server_edge_capacity + edge_agg_capacity + agg_core_capacity

    # init global network attrs
    init_global_graph_attrs(fat_tree_graph, 
                            max_nw_capacity, 
                            num_channels, 
                            node_labels=[ep_label,
                                         edge_label,
                                         aggregate_label,
                                         core_label],
                            topology_type='fat_tree')


    return fat_tree_graph


def init_global_graph_attrs(graph, max_nw_capacity, num_channels, topology_type='unknown', node_labels=['server']):
    '''
    Initialises the standard global graph attributes of a given graph

    Args:
    - graph (object): graph object
    - max_nw_capacity (int,float): max byte capacity of the graph
    '''
    graph.graph['num_channels_per_link'] = num_channels
    graph.graph['max_nw_capacity'] = max_nw_capacity
    graph.graph['curr_nw_capacity_used'] = 0
    graph.graph['num_active_connections'] = 0
    graph.graph['total_connections_blocked'] = 0
    graph.graph['node_labels'] = node_labels
    graph.graph['topology_type'] = topology_type
    graph.graph['channel_names'] = gen_channel_names(num_channels)


def gen_channel_names(num_channels):
    '''
    Generates channel names for channels on each link in network
    '''
    channels = [channel+1 for channel in range(num_channels)]
    channel_names = ['channel_' + str(channel) for channel in channels]
    
    return channel_names

def add_edge_capacity_attrs(graph, edge, channel_names, channel_capacity):
    '''
    Adds channels and corresponding max channel bytes to single edge in 
    graph
    
    Args:
    - graph (object): graph containing edges
    - edge (tuple): node-node pair in a tuple
    - channel_names (list of str): list of channel names to add to edge
    - channel_capacity (int, float): capacity to allocate to each channel
    '''
    attrs = {edge:
                {'channels': {channel: channel_capacity for channel in channel_names},
                 'max_channel_capacity': channel_capacity}}
    
    nx.set_edge_attributes(graph, attrs)




def add_edges_capacity_attrs(graph, 
                             edges,
                             channel_names,
                             channel_capacity):
    '''
    Adds channels and corresponding max channel capacitys to single edge in 
    graph
    
    To access e.g. the edge going from node 0 to node 1 (edge (0, 1)), you
    would index the graph with graph[0][1]

    To access e.g. the channel_1 attribute of this particular (0, 1) edge, you
    would do graph[0][1]['channels']['channel_1']
    
    Args:
    - graph (object): graph containing edges
    - edges (list of tuples): list of node pairs in tuples
    - channel_names (list of str): list of channel names to add to edge
    - channel_capacity (int, float): capacity to allocate to each channel
    '''
    attrs = {edge: 
                {'channels': 
                    {channel: channel_capacity for channel in channel_names},
                 'max_channel_capacity': 
                    channel_capacity
                 } for edge in edges}

    nx.set_edge_attributes(graph, attrs)


def plot_graph(graph, path_figure, name='network_graph.png', with_labels=True):
    plt.figure()
    nx.draw(graph, with_labels=with_labels, font_weight='bold')
    plt.savefig(path_figure + name)



if __name__ == '__main__':
    #graph = gen_simple_graph()
    #graph = gen_nsfnet_graph()
    graph = gen_fat_tree(k=3)
    
    plot_graph(graph, 'figures/graph/',name='network_graph.png',with_labels=True)
















    



