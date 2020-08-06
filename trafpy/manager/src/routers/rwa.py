import networkx as nx
import queue
import numpy as np

class RWA:
    
    def __init__(self, channel_names, num_k):
        self.channel_names = channel_names
        self.num_k = num_k
        self.blocking_table = np.array([[0, 0, 0]]) 

    def path_cost(self, graph, path, weight=None):
        '''
        Calculates cost of path. If no weight specified, 1 unit of cost is 1
        link/edge in the path

        Args:
        - path (list): list of node labels making up path from src to dst
        - weight (dict key): label of weight to be considered when evaluating
        path cost

        Returns:
        - pathcost (int, float): total cost of path
        '''
        pathcost = 0
        
        for i in range(len(path)):
            if i > 0:
                edge = (path[i-1], path[i])
                if weight != None:
                    pathcost += 1
                    # bugged: if in future want 1 edge cost != 1, fix this
                    #pathcost += graph[edge[0]][edge[1]][weight]
                else:
                    # just count the number of edges
                    pathcost += 1

        return pathcost

    def k_shortest_paths(self, graph, source, target, num_k=None, weight='weight'):
        '''
        Uses Yen's algorithm to compute the k-shortest paths between a source
        and a target node. The shortest path is that with the lowest pathcost,
        defined by external path_cost() function. Paths are returned in order
        of path cost, with lowest past cost being first etc.

        Args:
        - source (label): label of source node
        - target (label): label of destination node
        - num_k (int, float): number of shortest paths to compute
        - weight (dict key): dictionary key of value to be 'minimised' when
        finding 'shortest' paths

        Returns:
        - A (list of lists): list of shortest paths between src and dst
        '''
        if num_k is None:
            num_k = self.num_k

        # Shortest path from the source to the target
        A = [nx.shortest_path(graph, source, target, weight=weight)]
        A_costs = [self.path_cost(graph, A[0], weight)]

        # Initialize the heap to store the potential kth shortest path
        B = queue.PriorityQueue()

        for k in range(1, num_k):
            # spur node ranges first node to next to last node in shortest path
            try:
                for i in range(len(A[k-1])-1):
                    # Spur node retrieved from the prev k-shortest path, k - 1
                    spurNode = A[k-1][i]
                    # seq of nodes from src to spur node of prev k-shrtest path
                    rootPath = A[k-1][:i]

                    # We store the removed edges
                    removed_edges = []

                    for path in A:
                        if len(path) - 1 > i and rootPath == path[:i]:
                            # Remove edges of prev shrtest path w/ same root
                            edge = (path[i], path[i+1])
                            if not graph.has_edge(*edge):
                                continue
                            removed_edges.append((edge, graph.get_edge_data(*edge)))
                            graph.remove_edge(*edge)

                    # Calculate the spur path from the spur node to the sink
                    try:
                        spurPath = nx.shortest_path(graph, spurNode, target, weight=weight)

                        # Entire path is made up of the root path and spur path
                        totalPath = rootPath + spurPath
                        totalPathCost = self.path_cost(graph, totalPath, weight)
                        # Add the potential k-shortest path to the heap
                        B.put((totalPathCost, totalPath))

                    except nx.NetworkXNoPath:
                        pass

                    #Add back the edges that were removed from the graph
                    for removed_edge in removed_edges:
                        graph.add_edge(
                            *removed_edge[0],
                            **removed_edge[1]
                        )

                # Sort the potential k-shortest paths by cost
                # B is already sorted
                # Add the lowest cost path becomes the k-shortest path.
                while True:
                    try:
                        cost_, path_ = B.get(False)
                        if path_ not in A:
                            A.append(path_)
                            A_costs.append(cost_)
                            break
                    except queue.Empty:
                        break
            except IndexError:
                pass
        
        

        return A 


    def ff_k_shortest_paths(self, graph, k_shortest_paths, flow_size):
        '''
        Applies first fit algorithm, whereby path with lowest cost (shortest
        path) is looked at first when considering which route to select, then
        next etc. When route is considered, a wavelength (starting from lowest)
        is considered. If not available, move to next highest wavelength. If 
        go through all wavelengths and none available, move to next shortest
        path and try again. I.e. this is a 'select route first, then select
        wavelength' k-shortest path first fit RWA algorithm. If no routes- 
        wavelength pairs are available, message is blocked.

        Uses this first fit process to allocate a given demand a path and a 
        channel.

        Args:
        - k_shortest_paths (list of lists): the k shortest paths from the 
        source to destination node, with shortest path first etc
        - channel_names (list of strings): list of channel names that algorithm
        can consider assigning to each path
        - flow_size (int, float): size of demand

        Returns: 
        - path
        - channel
        '''
        print('Performing first fit....')
        for path in k_shortest_paths:
            print('Path considered: {}'.format(path))
            for channel in self.channel_names:
                path_edges = self.get_path_edges(path)
                print('Path edges: {}'.format(path_edges))
                if not self.check_if_channel_used(graph, path_edges, channel):
                    if self.check_if_channel_space(graph, path_edges, channel, round(flow_size,0)): 
                        return path, channel
                    else:
                        continue
                else:
                    continue
        
        # connection blocked
        path = ['N/A', 'N/A']
        channel = 'blocked'
        
        
        return path, channel

    def get_path_edges(self, path):
        '''
        Takes a path and returns list of edges in the path

        Args:
        - path (list): path in which you want to find all edges

        Returns:
        - edges (list of lists): all edges contained within the path
        '''
        num_nodes = len(path)
        num_edges = num_nodes - 1
        edges = [path[edge:edge+2] for edge in range(num_edges)]

        return edges

    def check_if_channel_used(self, graph, edges, channel):
        '''
        Takes list of edges to see if any one of the edges have used a certain
        channel

        Args:
        - edges (list of lists): edges we want to check if have used certain
        channel
        - channel (label): channel we want to check if has been used by any
        of the edges

        Returns:
        - True/False
        '''
        channel_used = False
        
        num_edges = len(edges)
        for edge in range(num_edges):
            node_pair = edges[edge]
            capacity = graph[node_pair[0]][node_pair[1]]['channels'][channel]
            if round(capacity,0) != round(graph[node_pair[0]][node_pair[1]]['max_channel_capacity'],0):
                channel_used = True
                break
            else:
                continue

        return channel_used

    def check_if_channel_space(self, graph, edges, channel, flow_size):
        '''
        Takes list of edges to see if all of the edges have enough space for
        the given demand on a certain channel

        Args:
        - edges (list of lists): edges we want to check if have enough space
        for a certain demand on a certain channel
        - channel (label): channel we want to check if has enough space
        for the given demand across all given edges
        - flow_size: demand size we want to check if there's space for on 
        the given channel across all given edges


        Returns:
        - True/False
        '''
        channel_space = True
        
        num_edges = len(edges)
        for edge in range(num_edges):
            node_pair = edges[edge]
            capacity = graph[node_pair[0]][node_pair[1]]['channels'][channel]
            if capacity - flow_size < 0:
                channel_space = False
            else:
                continue

        return channel_space
        

    def get_action(self, observation):
        '''
        Gets an action (route+channel or blocked) for DCN simulation
        '''
        src = observation['pair'][0]
        dst = observation['pair'][1]
        establish = observation['establish']
        flow_size = observation['flow_size']
        network_state = observation['network_state']
        k_shortest_paths = self.k_shortest_paths(network_state, src, dst)
        
        if establish == 1:
            # need to establish connection
            path, channel = self.ff_k_shortest_paths(network_state, k_shortest_paths, flow_size)
        else:
            # just a take down request
            path = [src,dst]
            channel = 'N/A'

        action = {'path': path,
                  'channel': channel,
                  'flow_size': flow_size,
                  'establish': establish,
                  'k_shortest_paths': k_shortest_paths}

        return action
            
















