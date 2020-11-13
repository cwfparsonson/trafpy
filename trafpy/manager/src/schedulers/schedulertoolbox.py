import numpy as np
import networkx as nx
import copy
import math
import sys
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import itertools


class SchedulerToolbox:

    def __init__(self, Graph, RWA, slot_size):
        self.Graph = Graph
        self.RWA = RWA
        self.slot_size = slot_size
        
        # self.packet_size = 3000
        self.packet_size = 300
        
        self.reset()

    def reset(self):
        self.SchedulerNetwork = copy.deepcopy(self.Graph)
        
    def remove_flow_from_queue(self, flow_dict, network):
        '''
        Given flow dict and network that flow is in, will locate flow 
        in network and remove from queue
        '''
        
        sn = flow_dict['src']
        dn = flow_dict['dst']
        queued_flows = network.nodes[sn][dn]['queued_flows']
        idx = self.find_flow_idx(flow_dict, queued_flows)
        del network.nodes[sn][dn]['queued_flows'][idx]
        del network.nodes[sn][dn]['completion_times'][idx]

        return network

    def filter_unavailable_flows(self, network):
        '''
        Takes a network and filters out any flow that is not ready to be scheduled
        yet i.e. has incomplete parent flow dependencies. Use this method to get
        network representation for 'job-agnostic' flow scheduling systems.
        '''
        net = copy.deepcopy(network)
        
        eps = network.graph['endpoints']
        for ep in eps:
            ep_queues = network.nodes[ep]
            for ep_queue in ep_queues.values():
                for flow_dict in ep_queue['queued_flows']:
                    if flow_dict['can_schedule'] == 0:
                        # can't schedule, filter out of network
                        net = self.remove_flow_from_queue(flow_dict, net)
                    else:
                        # can schedule
                        pass
                        # can schedule
        
        # check no bad flows left in queue
        for ep in eps:
            ep_queues = net.nodes[ep]
            for ep_queue in ep_queues.values():
                for flow_dict in ep_queue['queued_flows']:
                    if flow_dict['can_schedule'] == 0:
                        sys.exit('Illegal flow(s) still present')
                    else:
                        pass

        
        
        return net
        
        
    def update_network_state(self, observation, hide_child_dependency_flows=True):
        '''
        If hide_child_dependency_flows is True, will only update scheduler network
        to see flows that are ready to be scheduled i.e. all parent flow dependencies
        have been completed. This is used for 'job-agnostic' scheduling systems
        which, rather than considering the job that each flow is part of, only consider
        the flow.
        
        If False, will just update network with all flows (even those that cannot yet
        be scheduled). This is used for 'job- & network- aware' scheduling systems.
        '''
        if hide_child_dependency_flows:
            self.SchedulerNetwork = self.filter_unavailable_flows(observation['network'])
        else: 
            self.SchedulerNetwork = copy.deepcopy(observation['network'])
            
            
            
    
    def gen_flow_packets(self, flow_size):
        num_packets = math.ceil(flow_size/self.packet_size) # round up 
        packets = (np.ones(num_packets)) * self.packet_size
        return self.packet_size, packets
    
    def get_flow_from_network(self, server, queue, flow_idx):
        '''
        Given the server that the flow is at, the queue of the server that
        the flow is in, and the flow idx of the flow in the queue, this method
        returns the flow dict
        '''
        try:
            return self.SchedulerNetwork.nodes[server][queue]['queued_flows'][flow_idx]
        except (KeyError, IndexError):
            # flow being searched doesn't exist
            return 'N/A'


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
    
    def find_flow_idx(self, flow, flows):
        '''
        Finds flow idx in a list of flows. Assumes the following 
        flow features are unique and unchanged properties of each flow:
        - flow size
        - source
        - destination
        - time arrived

        Args:
        - flow (dict): flow dictionary
        - flows (list of dicts) list of flows in which to find flow idx
        '''
        size = flow['size']
        src = flow['src']
        dst = flow['dst']
        time_arrived = flow['time_arrived']

        idx = 0
        for f in flows:
            if f['size']==size and f['src']==src and f['dst']==dst and f['time_arrived']==time_arrived:
                # flow found in flows
                return idx
            else:
                # flow not found, move to next f in flows
                idx += 1
        
        return sys.exit('Flow not found in list of flows')

    def find_flow_queue(self, flow):
        '''
        Finds queue of flow in network
        '''
        sn = flow['src']
        dn = flow['dst']
        flow_queue = self.SchedulerNetwork.nodes[sn][dn]['queued_flows']

        return flow_queue


    def check_if_lightpath_available(self, edges, channel, chosen_flows):
        '''
        Checks if chosen flow already has edges which have been assigned to 
        channel
        '''
        taken_paths = [flow['path'] for flow in chosen_flows]
        taken_channels = [flow['channel'] for flow in chosen_flows]
        for taken_path in taken_paths:
            taken_edges = self.get_path_edges(taken_path)
            if any(e in taken_edges for e in edges) or any(e[::-1] in taken_edges for e in edges):
                # prev chosen paths contains 1 or more edges of curr chosen path
                # check if taken path assigned to chosen channel
                idx = taken_paths.index(taken_path)
                if chosen_flows[idx]['channel'] == channel:
                    # chosen channel already assigned to edge(s) in chosen path
                    # lightpath not available
                    return False
        return True
    
    def estimate_time_to_completion(self, flow_dict):
        num_packets = len(flow_dict['packets'])
        try:
            packet_size = flow_dict['packets'][0]
        except IndexError:
            print('Flow has 0 packets since has 0 size. Flows should not have 0 size by definition. This usually results from not setting a minimum bound on the possible values when generating your flow size distribution.')
            sys.exit()
        
        path_links = self.get_path_edges(flow_dict['path'])
        link_bws = []
        for link in path_links:
            link_bws.append(self.SchedulerNetwork[link[0]][link[1]]['max_channel_capacity'])
        lowest_bw = min(link_bws)
        
        size_per_slot = lowest_bw/(1/self.slot_size)
        packets_per_slot = int(size_per_slot / packet_size) # round down 
        if packets_per_slot == 0:
            raise Exception('Encountered 0 packets that can be transferred per time slot. Either decrease packet size or increase time slot size.')
        slots_to_completion = math.ceil(num_packets/packets_per_slot) # round up
        completion_time = slots_to_completion * self.slot_size
        
        return completion_time, packets_per_slot

    def binary_encode_time_in_queue(self, time_in_queue, max_record_time_in_queue):
        num_decimals = str(self.slot_size)[::-1].find('.')
        times = [np.round(t, num_decimals) for t in np.arange(0,
                                                              max_record_time_in_queue+self.slot_size,
                                                              self.slot_size)]
        time_to_int = dict( (t,i) for i,t in enumerate(times) )
        
        binary_encoded_time_in_queue = [0 for _ in range(int(len(times)))]
        
        round_to_nearest = self.slot_size
        factor = 1 / round_to_nearest
        rounded_time_in_queue = math.floor(time_in_queue * factor) / factor
        if rounded_time_in_queue > max_record_time_in_queue:
            rounded_time_in_queue = max_record_time_in_queue
        idx = time_to_int[rounded_time_in_queue]
        binary_encoded_time_in_queue[idx] = 1

        return binary_encoded_time_in_queue

    def binary_encode_num_packets(self, num_packets, max_record_num_packets):
        binary_encoded_num_packets = [0 for _ in range(max_record_num_packets+1)]
        if num_packets > max_record_num_packets:
            num_packets = num_packets
        binary_encoded_num_packets[num_packets] = 1
        
        return binary_encoded_num_packets

    def binary_encode_num_flows_in_queue(self, num_flows_in_queue, max_num_flows_in_queue):
        binary_encoded_num_flows = [0 for _ in range(max_num_flows_in_queue+1)]
        if num_flows_in_queue > max_num_flows_in_queue:
            num_flows_in_queue = max_num_flows_in_queue
        binary_encoded_num_flows[num_flows_in_queue] = 1

        return binary_encoded_num_flows

    def binary_encode_paths(self, paths):
        graph_edges = [e for e in self.SchedulerNetwork.edges]

        # map edges to integers
        edge_to_int = dict((e,i) for i,e in enumerate(graph_edges))
        int_to_edge = dict((i,e) for i,e in enumerate(graph_edges))

        # integer encode paths
        int_encoded_paths = []
        for path in paths:
            path_edges = self.get_path_edges(path)
            path_edges = [tuple(e) for e in path_edges]    
           
            encoded_path = []
            for edge in path_edges:
                try:
                    encoded_path.append(edge_to_int[edge])
                except KeyError:
                    # undirected graph therefore can flip edge order
                    encoded_path.append(edge_to_int[edge[::-1]])
            
            int_encoded_paths.append(encoded_path)

        # binary encode paths
        binary_encoded_paths = []
        for int_path in int_encoded_paths:
            binary_encoded_path = [0 for _ in range(len(graph_edges))]
            for idx in int_path:
                binary_encoded_path[idx] = 1
            binary_encoded_paths.append(binary_encoded_path)

        return binary_encoded_paths

    def init_paths_and_packets(self, flow_dict):
        if flow_dict['k_shortest_paths'] is None:
            k_shortest_paths = self.RWA.k_shortest_paths(self.SchedulerNetwork,
                                                         flow_dict['src'],
                                                         flow_dict['dst'])
            flow_dict['k_shortest_paths'] = k_shortest_paths
        else:
            # previously calculated and saved
            pass
        
        if flow_dict['packets'] is None:
            _, packets = self.gen_flow_packets(flow_dict['size'])
            flow_dict['packets'] = packets
        else:
            # previously calculated and saved
            pass

        return flow_dict


    def find_shortest_flow_in_queue(self, queued_flows, completion_times):
        '''
        Allocates shortest lightpaths and finds flow in queue w/ shortest 
        completion time
        '''
        num_queued_flows = len(queued_flows)
        for flow_idx in range(num_queued_flows):
            flow_dict = self.init_paths_and_packets(queued_flows[flow_idx])
            #shortest_paths = flow_dict['k_shortest_paths']
            #size = flow_dict['size']
            #path, _ = self.RWA.ff_k_shortest_paths(self.SchedulerNetwork,shortest_paths,size) 
            path = flow_dict['k_shortest_paths'][0]
            channel = self.RWA.channel_names[0]
            flow_dict['path'] = path
            flow_dict['channel'] = channel
            completion_time, _ = self.estimate_time_to_completion(flow_dict)
            completion_times[flow_idx] = completion_time
        comp_time = min(completion_times)
        active_flow_idx = completion_times.index(comp_time)
        shortest_flow = queued_flows[active_flow_idx]
        
        return shortest_flow, comp_time
    
    def get_curr_queue_states(self, network):
        '''
        Returns all queues (empty and full) in network
        '''

        N = int(len(self.Graph.graph['endpoints'])) # number of servers
        num_queues = N * (N - 1)
        
        queues = {ep: None for ep in self.Graph.graph['endpoints']}
        
        for ep in self.Graph.graph['endpoints']:
            ep_queues = network.nodes[ep]
            queues[ep] = ep_queues
        
        return queues

    def look_for_available_lightpath(self, chosen_flow, chosen_flows, search_k_shortest=True):
        '''
        If search_k_shortest, will look at all k shortest paths available. 
        in flow['k_shortest_paths']. If False, will only consider flow['path'] 
        already assigned.

        '''
        # check for link contentions
        if search_k_shortest:
            paths = chosen_flow['k_shortest_paths']
        else:
            paths = [chosen_flow['path']]
        channels = self.RWA.channel_names
        taken_paths = [flow['path'] for flow in chosen_flows]
        taken_edges = [self.get_path_edges(taken_path) for taken_path in taken_paths]
        taken_channels = [flow['channel'] for flow in chosen_flows]
        establish_flow = False
        # check if any lightpath available without needing to take down connection
        for path in paths:
            for channel in channels:
                edges = self.get_path_edges(path)
                if self.check_if_lightpath_available(edges, channel, chosen_flows):
                    establish_flow = True
                    break
            else:
                continue
            break

        return establish_flow, path, channel
