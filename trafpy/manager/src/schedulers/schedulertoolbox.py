import numpy as np
import networkx as nx
import copy
import math
import sys
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import itertools
import json
import random


class SchedulerToolbox:

    def __init__(self, Graph, RWA, slot_size, packet_size=300, time_multiplexing=True, debug_mode=False):
        self.Graph = Graph
        self.RWA = RWA
        self.slot_size = slot_size
        self.packet_size = packet_size
        self.time_multiplexing = time_multiplexing
        self.debug_mode = debug_mode
        
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

    def filter_unavailable_flows(self):
        '''
        Takes a network and filters out any flow that is not ready to be scheduled
        yet i.e. has incomplete parent flow dependencies. Use this method to get
        network representation for 'job-agnostic' flow scheduling systems.
        '''
        net = self.SchedulerNetwork
        
        eps = net.graph['endpoints']
        for ep in eps:
            ep_queues = net.nodes[ep]
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
        
    def reset_channel_capacities_of_edges(self):
        '''Takes edges and resets their available capacities back to their maximum capacities.'''
        net = self.SchedulerNetwork
        for edge in net.edges:
            for channel in net[edge[0]][edge[1]]['channels']:
                # reset channel capacity
                init_chancap = net[edge[0]][edge[1]]['channels'][channel]
                net[edge[0]][edge[1]]['channels'][channel] = net[edge[0]][edge[1]]['max_channel_capacity']
                reset_chancap = net[edge[0]][edge[1]]['channels'][channel]
        # update global graph property
        net.graph['curr_nw_capacity_used'] = 0


        
    def update_network_state(self, observation, hide_child_dependency_flows=True, reset_channel_capacities=True):
        '''
        If hide_child_dependency_flows is True, will only update scheduler network
        to see flows that are ready to be scheduled i.e. all parent flow dependencies
        have been completed. This is used for 'job-agnostic' scheduling systems
        which, rather than considering the job that each flow is part of, only consider
        the flow.
        
        If False, will just update network with all flows (even those that cannot yet
        be scheduled). This is used for 'job- & network- aware' scheduling systems.
        '''
        self.SchedulerNetwork = copy.deepcopy(observation['network'])

        if reset_channel_capacities:
            self.reset_channel_capacities_of_edges()
        if hide_child_dependency_flows:
            self.filter_unavailable_flows()

            
            
            
    
    def gen_flow_packets(self, flow_size):
        num_packets = math.ceil(flow_size/self.packet_size) # round up 
        # packets = [self.packet_size for _ in range(num_packets)]

        return self.packet_size, num_packets
    
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

    def set_up_connection(self, flow, num_decimals=6):
        '''
        Sets up connection between src-dst node pair by removing capacity from
        all edges in path connecting them. Also updates graph's global curr 
        network capacity used property
        
        Args:
        - flow (dict): flow dict containing flow info to set up
        '''
        if self.debug_mode:
            print('Setting up connection for flow {}'.format(flow))
        path = flow['path']
        channel = flow['channel']
        flow_size = flow['size']
        packet_size = flow['packet_size']
        packets = flow['packets']
        packets_this_slot = flow['packets_this_slot']

        info_to_transfer_this_slot = packets_this_slot * packet_size
        capacity_used_this_slot = round(info_to_transfer_this_slot / self.slot_size, num_decimals) # info units of this flow transferred this time slot == capacity used on each channel in flow's path this time slot

        edges = self.get_path_edges(path)

        num_edges = len(edges)
        for edge in range(num_edges):
            node_pair = edges[edge]
            # update edge capacity remaining after establish this flow
            init_bw = self.get_channel_bandwidth(node_pair, channel)
            self.SchedulerNetwork[node_pair[0]][node_pair[1]]['channels'][channel] -= capacity_used_this_slot
            self.SchedulerNetwork[node_pair[0]][node_pair[1]]['channels'][channel] = round(self.SchedulerNetwork[node_pair[0]][node_pair[1]]['channels'][channel], num_decimals)
            if self.debug_mode:
                print('Updated {} capacity for {}: {} -> {}'.format(channel, node_pair, init_bw, self.get_channel_bandwidth(node_pair, channel)))

            # check that establishing this flow is valid given edge capacity constraints
            if self.SchedulerNetwork[node_pair[0]][node_pair[1]]['channels'][channel] < 0:
                raise Exception('Tried to set up flow {} on edge {} channel {}, but this results in a negative channel capacity on this edge i.e. this edge\'s channel is full, cannot have more flow packets scheduled!'.format(flow, edge, channel)) 

            # update global graph property
            self.SchedulerNetwork.graph['curr_nw_capacity_used'] += capacity_used_this_slot
        self.SchedulerNetwork.graph['num_active_connections'] += 1

    def take_down_connection(self, flow, num_decimals=6):
        '''
        Removes established connection by adding capacity back onto all edges
        in the path connecting the src-dst node pair. Also updates graph's
        global curr network capacity used property

        Args:
        - flow (dict): flow dict containing info of flow to take down
        '''
        if self.debug_mode:
            print('Taking down connection for flow {}'.format(flow))
        path = flow['path']
        channel = flow['channel']
        flow_size = flow['size']
        packet_size = flow['packet_size']
        packets = flow['packets']
        packets_this_slot = flow['packets_this_slot']

        info_to_transfer_this_slot = packets_this_slot * packet_size
        capacity_used_this_slot = round(info_to_transfer_this_slot / self.slot_size, num_decimals) # info units of this flow transferred this time slot == capacity used on each channel in flow's path this time slot

        edges = self.get_path_edges(path)
        
        num_edges = len(edges)
        for edge in range(num_edges):
            node_pair = edges[edge]
            # update edge property
            init_bw = self.get_channel_bandwidth(node_pair, channel)
            self.SchedulerNetwork[node_pair[0]][node_pair[1]]['channels'][channel] += capacity_used_this_slot
            self.SchedulerNetwork[node_pair[0]][node_pair[1]]['channels'][channel] = round(self.SchedulerNetwork[node_pair[0]][node_pair[1]]['channels'][channel], num_decimals)
            if self.debug_mode:
                print('Updated {} capacity for {}: {} -> {}'.format(channel, node_pair, init_bw, self.get_channel_bandwidth(node_pair, channel)))
            # update global graph property
            self.SchedulerNetwork.graph['curr_nw_capacity_used'] -= capacity_used_this_slot
        self.SchedulerNetwork.graph['num_active_connections'] -= 1

    def get_channel_bandwidth(self, edge, channel):
        '''Gets current channel bandwidth left on a given edge in the network.'''
        try:
            return self.SchedulerNetwork[edge[0]][edge[1]]['channels'][channel]
        except KeyError:
            return self.SchedulerNetwork[edge[1]][edge[0]]['channels'][channel]

    def check_if_lightpath_available(self, path, channel, chosen_flows):
        '''
        Checks if chosen flow already has edges which have been assigned to 
        channel. If it does and if time_multiplexing, checks if any space left
        on contentious path channels.
        '''
        edges = self.get_path_edges(path)
        taken_paths = [flow['path'] for flow in chosen_flows]
        taken_channels = [flow['channel'] for flow in chosen_flows]
        for taken_path in taken_paths:
            taken_edges = self.get_path_edges(taken_path)
            # if any(e in taken_edges for e in edges) or any(e[::-1] in taken_edges for e in edges):
            for e in edges:
                if e in taken_edges or e[::-1] in taken_edges:
                    # prev chosen paths contains 1 or more edges of curr chosen path
                    # check if taken path assigned to chosen channel
                    idx = taken_paths.index(taken_path)
                    if chosen_flows[idx]['channel'] == channel:
                        # chosen channel already assigned to edge(s) in chosen path
                        if not self.time_multiplexing:
                            # can only have 1 flow per channel per edge, lightpath not available
                            return False
                        else:
                            # check if any bandwidth available
                            available_bandwidth = self.get_channel_bandwidth(e, channel)
                            # _, max_packets_per_slot = self.get_max_flow_info_transferred_per_slot(chosen_flows[idx], path, channel)
                            if available_bandwidth != 0:
                                # can schedule
                                return True
                            else:
                                # cannot schedule any more packets this time slot
                                return False

        return True
    
    def estimate_time_to_completion(self, flow_dict):
        # num_packets = len(flow_dict['packets'])
        num_packets = flow_dict['packets']
        packet_size = flow_dict['packet_size']
        
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

    # def binary_encode_num_packets(self, num_packets, max_record_num_packets):
        # binary_encoded_num_packets = [0 for _ in range(max_record_num_packets+1)]
        # if num_packets > max_record_num_packets:
            # num_packets = num_packets
        # binary_encoded_num_packets[num_packets] = 1
        
        # return binary_encoded_num_packets

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
            _, num_packets = self.gen_flow_packets(flow_dict['size'])
            flow_dict['packets'] = num_packets
            flow_dict['packet_size'] = self.packet_size
        else:
            # previously calculated and saved
            pass

        return flow_dict

    def get_max_flow_info_transferred_per_slot(self, flow, path, channel):
        '''
        Returns maximum possible flow information & number of packets transferred
        per timeslot given the flow's path (i.e. in point-to-point circuit switched
        network, max info transferred per slot is the bandwidth of the lowest bw
        link in the path * the slot size)
        '''
        packet_size = flow['packet_size']
        path_links = self.get_path_edges(path)
        link_bws = []
        for link in path_links:
            link_bws.append(self.get_channel_bandwidth(link, channel))
        capacity = min(link_bws) # channel capacity == info transferred per unit time
        info_per_slot = capacity  * self.slot_size # info transferred per slot == info transferred per unit time * number of time units (i.e. slot size)
        packets_per_slot = int(info_per_slot / packet_size) # round down 

        return info_per_slot, packets_per_slot

    def choose_num_packets_this_time_slot(self, flow, path, channel):
        '''
        Given a flow, path and channel, finds the number of packets that this flow
        can have scheduled this time slot given the channel capacities of the link
        channels in the path.

        '''
        max_info_per_slot, max_packets_per_slot = self.get_max_flow_info_transferred_per_slot(flow, path, channel)
        if max_packets_per_slot > flow['packets']:
            # fewer packets than maximum allowed per time slot, can schedule fewer packets
            num_packets = flow['packets']
        else:
            # just schedule maximum packets this time slot
            num_packets = max_packets_per_slot
        return num_packets
        





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
            flow_dict['packets_this_slot'] = self.choose_num_packets_this_time_slot(flow_dict, path, channel)
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

    def look_for_available_lightpath(self, chosen_flow, chosen_flows, search_k_shortest=True, random_shuffle=False):
        '''
        If search_k_shortest, will look at all k shortest paths available
        in flow['k_shortest_paths']. If False, will only consider flow['path'] 
        already assigned.

        If random_shuffle, will randomly shuffle paths and channels before searching through them.

        '''
        # check for link contentions
        if search_k_shortest:
            paths = chosen_flow['k_shortest_paths']
        else:
            paths = [chosen_flow['path']]
        channels = self.RWA.channel_names
        if random_shuffle:
            random.shuffle(paths)
            random.shuffle(channels)

        taken_paths = [flow['path'] for flow in chosen_flows]
        taken_edges = [self.get_path_edges(taken_path) for taken_path in taken_paths]
        taken_channels = [flow['channel'] for flow in chosen_flows]
        establish_flow = False
        packets_this_slot = 0
        # check if any lightpath available without needing to take down connection
        for path in paths:
            for channel in channels:
                if self.check_if_lightpath_available(path, channel, chosen_flows):
                    _, max_packets_per_slot = self.get_max_flow_info_transferred_per_slot(chosen_flow, path, channel)
                    if max_packets_per_slot != 0:
                        establish_flow = True
                        packets_this_slot = self.choose_num_packets_this_time_slot(chosen_flow, path, channel)
                        break
            else:
                continue
            break

        return establish_flow, path, channel, packets_this_slot


    def calc_basrpt_cost(self, flow, V, N):
        flow_queue = self.find_flow_queue(flow)
        num_queued_flows = len(flow_queue)

        # calc size of flow's queue
        queue_length = 0
        for flow_idx in range(num_queued_flows):
            flow_dict = flow_queue[flow_idx]
            if flow_dict['packets'] is None:
                queued_flow_bytes = flow_dict['size']
            else:
                queued_flow_bytes = flow_dict['packets'] * flow_dict['packet_size']
            queue_length += queued_flow_bytes

        # calc flow fct
        fct, _ = self.estimate_time_to_completion(flow)

        # calc cost
        cost = ((V/N)*fct) - queue_length

        return cost

    def find_all_contending_flows(self, chosen_flow, chosen_flows, cost_metric, **kwargs):
        # check cost metric given is valid
        if cost_metric == 'basrpt_cost':
            required_kwargs = ['V', 'N']

        for kwarg in kwargs.keys():
            if kwarg not in required_kwargs:
                raise Exception('{} metric requires kwargs {} but have only given {}'.format(cost_metric, required_kwargs, kwargs))
        
        


        contending_flows = {}

        # 'prospective' chosen flow params
        paths = chosen_flow['k_shortest_paths']
        channels = self.RWA.channel_names
        
        # 'already chosen' flows params
        taken_paths = [flow['path'] for flow in chosen_flows]
        taken_edges = [self.get_path_edges(taken_path) for taken_path in taken_paths]
        taken_channels = [flow['channel'] for flow in chosen_flows]

        # find contention(s) between prospective chosen_flow and already chosen flows
        # i.e. find all already chosen flows whose established path and channel might possibly
        # contend with the prospective chosen flow's path and channel
        found_contention=False
        for path in paths:
            edges = self.get_path_edges(path)
            for channel in channels:
                for idx in range(len(taken_paths)):
                    taken_path_edges = taken_edges[idx]
                    taken_channel = taken_channels[idx]
                    if taken_channel == channel:
                        for e in edges:
                            if e in taken_path_edges or e[::-1] in taken_path_edges:
                                # check if any bandwidth available on this edge channel
                                available_bandwidth = self.get_channel_bandwidth(e, channel)
                                if self.time_multiplexing:
                                    # can have as many flows per channel per edge as bandwidth allows
                                    if available_bandwidth == 0:
                                        # no more bandwidth on this channel, contention
                                        found_contention=True
                                else:
                                    # can only have 1 flow per channel per edge
                                    found_contention=True
                    if found_contention:
                        if channel not in contending_flows.keys():
                            # have not yet encountered this channel, init channel dict -> register contending flow
                            contending_flows[channel] = {}
                            contending_flows[channel]['cont_f'] = [chosen_flows[idx]]
                            contending_flows[channel]['chosen_p'] = [path]
                            contending_flows[channel]['chosen_c'] = [channel]
                            contending_flows[channel]['chosen_num_packets'] = [chosen_flows[idx]['packets_this_slot']]
                            if cost_metric == 'basrpt_cost':
                                cost = self.calc_basrpt_cost(chosen_flows[idx], kwargs['V'], kwargs['N'])
                            elif cost_metric == 'fct':
                                cost, _ = self.estimate_time_to_completion(chosen_flows[idx])
                            contending_flows[channel]['cost'] = [cost]

                        else:
                            # register contending flow
                            contending_flows[channel]['cont_f'].append(chosen_flows[idx])
                            contending_flows[channel]['chosen_p'].append(path)
                            contending_flows[channel]['chosen_c'].append(channel)
                            contending_flows[channel]['chosen_num_packets'].append(chosen_flows[idx]['packets_this_slot'])
                            if cost_metric == 'basrpt_cost':
                                cost = self.calc_basrpt_cost(chosen_flows[idx], kwargs['V'], kwargs['N'])
                            elif cost_metric == 'fct':
                                cost, _ = self.estimate_time_to_completion(chosen_flows[idx])
                            contending_flows[channel]['cost'].append(cost)

                        found_contention = False # reset for next contention check
        if len(contending_flows.keys()) == 0:
            raise Exception('Could not find where any contentions occurred')

        return contending_flows
        

    def choose_channel_and_path_using_contending_flows(self, contending_flows):
        # find initial guess for contending flow to choose a path and channel
        # find which channel has least contentious (highest cost) most contentious flow
        # -> chosen flow will try to beat contending flows on this 'easiest to beat' channel
        highest_cost_by_channel = {channel: max(contending_flows[channel]['cost']) for channel in contending_flows.keys()}
        chosen_channel = min(highest_cost_by_channel, key=highest_cost_by_channel.get)

        # find which of the contending flows for this channel is the most contentious (i.e. has lowest cost)
        # if chosen flow can beat this flow, it will beat all other flows on this channel and therefore should replace them all
        idx_min_cost = contending_flows[chosen_channel]['cost'].index(min(contending_flows[chosen_channel]['cost']))
        contending_flow = contending_flows[chosen_channel]['cont_f'][idx_min_cost]
        contending_flow_cost = contending_flows[chosen_channel]['cost'][idx_min_cost]
        chosen_path = contending_flows[chosen_channel]['chosen_p'][idx_min_cost]

        # group contending flows on chosen channel -> these are the contending flows which
        # should be considered when determining whether chosen flow should be established
        contending_flows_list = contending_flows[chosen_channel]['cont_f']

        # remove duplicate contending flows (since in above loop, added contending flow for each k path)
        _list = []
        for f in contending_flows_list:
            if f not in _list:
                _list.append(f)
        contending_flows_list = _list

        return chosen_path, chosen_channel, contending_flow, contending_flow_cost, contending_flows_list



    def get_packets_available_if_drop_all_contending_flows(self, chosen_flow, chosen_path, chosen_channel, contending_flows_list):
        # if all contending packets dropped, find num packets that would be made available to the chosen flow
        chosen_edges = self.get_path_edges(chosen_path)
        chosen_edge_packets_available = {json.dumps(e): (self.slot_size*self.get_channel_bandwidth(e,chosen_channel))/chosen_flow['packet_size'] for e in chosen_edges} # track number of packets made available on each chosen path's edge were all contending flows to be dropped
        for flow in contending_flows_list:
            edges_freed = self.get_path_edges(flow['path']) # edges freed by dropping this contending flow
            packets_freed = flow['packets_this_slot'] # packets freed by dropping this contending flow
            for e in edges_freed:
                if json.dumps(e) in chosen_edge_packets_available.keys():
                    chosen_edge_packets_available[json.dumps(e)] += packets_freed
                else:
                    try:
                        chosen_edge_packets_available[json.dumps(e[::-1])] += packets_freed
                    except KeyError:
                        # this edge of contending flow is not in chosen flow's path, ignore
                        pass
        packets_freed = min(chosen_edge_packets_available.values())

        return packets_freed

    def get_packets_available_outside_contending_edges(self, chosen_flow, chosen_path, chosen_channel, contending_flows_list):
        # # find number of packets available to chosen flow for any of the chosen path's edges which are not an edge of a contending flow

        # init list of contending flow edges
        contending_flows_edges = []
        for flow in contending_flows_list:
            for e in self.get_path_edges(flow['path']):
                contending_flows_edges.append(e)

        chosen_edges = self.get_path_edges(chosen_path)
        edge_bws = [] # track edge bandwidths outside contending flows
        for e in chosen_edges:
            if e in contending_flows_edges or e[::-1] in contending_flows_edges:
                # edge already considered w.r.t. at least one contending flow
                pass
            else:
                # edge not yet considered by only considering contending flows
                edge_bws.append(self.get_channel_bandwidth(e, chosen_channel))
        if len(edge_bws) == 0:
            # no other edges other than those in the contending flows need be considered when considering what the limiting available bandwidth is in the chosen path
            lowest_packets_available_outside_contending_edges = None
        else:
            lowest_bw = min(edge_bws)
            lowest_info_per_slot = lowest_bw * self.slot_size
            lowest_packets_available_outside_contending_edges = int(lowest_info_per_slot / chosen_flow['packet_size'])

        return lowest_packets_available_outside_contending_edges

    def get_maximum_packets_available_if_all_edges_empty(self, chosen_flow, chosen_path):
        # get lowest maximum channel capacity for chosen path edges -> this is maximum possible rate of information for flow assuming all other contending flows dropped
        max_chan_caps = []
        for edge in self.get_path_edges(chosen_path):
            for channel in self.SchedulerNetwork[edge[0]][edge[1]]['channels']:
                max_chan_caps.append(self.SchedulerNetwork[edge[0]][edge[1]]['max_channel_capacity'])
        max_chan_cap = min(max_chan_caps)
        max_packets = (self.slot_size * max_chan_cap) / chosen_flow['packet_size'] # max poss num packets

        return max_packets

    def get_maximum_packets_requestable_by_flow(self, chosen_flow, max_packets_available_if_all_edges_empty, packets_available_outside_contending_edges):
        # find max packets that chosen flow could possibly request on contending edges given 1. its size and 2. channel availability outside of the contending flows edges
        if packets_available_outside_contending_edges is None:
            # no other edges other than contending flows edges need be considered, can request up to max packets (or just all remaining packets of flow)
            return min(chosen_flow['packets'], max_packets_available_if_all_edges_empty)
        else:
            # must also consider edges outside contending flows in determing maximum number of packets the flow can request on contending edges
            return min(chosen_flow['packets'], packets_available_outside_contending_edges, max_packets_available_if_all_edges_empty)



    def select_minimum_number_of_contending_flows_to_drop(self, chosen_flow, chosen_path, chosen_channel, contending_flows_list, max_packets_requested_by_chosen_flow, max_packets_available_if_all_edges_empty, cost_metric, **kwargs):
        # chosen flow cannot request the number of packets that would be freed up by dropping all found contending flows, do not need to drop all contending flow it would be inefficient to drop all flows, therefore should drop minimum number of
        # contentious flows needed until enough bandwidth is freed
        # check cost metric given is valid
        if cost_metric == 'basrpt_cost':
            required_kwargs = ['V', 'N']

        for kwarg in kwargs.keys():
            if kwarg not in required_kwargs:
                raise Exception('{} metric requires kwargs {} but have only given {}'.format(cost_metric, required_kwargs, kwargs))

        # find flows which need to be dropped to free enough space
        packets_available_if_drop_all_contending_flows = 0
        contending_flows_corrected = []
        chosen_edges = self.get_path_edges(chosen_path)
        chosen_edge_packets_available = {json.dumps(e): (self.slot_size*self.get_channel_bandwidth(e,chosen_channel))/chosen_flow['packet_size'] for e in chosen_edges}
        packets_freed_list = []
        while True:
            # drop flows until have minimum amount bandwidth/packets needed
            # on all edges in chosen path
            # choose the least contentious (highest cost) flow to drop
            if cost_metric == 'basrpt_cost':
                costs = [self.calc_basrpt_cost(f, kwargs['V'], kwargs['N']) for f in contending_flows_list]
            elif cost_metric == 'fct':
                costs = []
                for f in contending_flows_list:
                    cost, _ = self.estimate_time_to_completion(f)
                    costs.append(cost)

            idx_max_cost = costs.index(max(costs))
            contending_flows_corrected.append(contending_flows_list[idx_max_cost])
            edges_freed = self.get_path_edges(contending_flows_list[idx_max_cost]['path'])
            packets_freed = contending_flows_list[idx_max_cost]['packets_this_slot']
            packets_available_if_drop_all_contending_flows += packets_freed
            packets_freed_list.append(packets_freed)
            for e in edges_freed:
                if json.dumps(e) in chosen_edge_packets_available.keys():
                    chosen_edge_packets_available[json.dumps(e)] += packets_freed
                else:
                    try:
                        chosen_edge_packets_available[json.dumps(e[::-1])] += packets_freed
                    except KeyError:
                        # this edge of contending flow is not in chosen flow's path, ignore
                        pass

            del contending_flows_list[idx_max_cost]
            # print('Packets requested: {} | Packets freed: {} | chosen_edge_packets_available:\n{}'.format(max_packets_requested_by_chosen_flow, packets_available_if_drop_all_contending_flows, chosen_edge_packets_available))
            if packets_available_if_drop_all_contending_flows >= max_packets_requested_by_chosen_flow and min(chosen_edge_packets_available.values()) >= max_packets_requested_by_chosen_flow:
                break
            else:
                continue
        
        # update contending flows & packets made available by dropping them
        packets_scheduled_if_drop_all_contending_flows = min(max_packets_available_if_all_edges_empty, packets_available_if_drop_all_contending_flows)
        contending_flows_list = contending_flows_corrected
        if cost_metric == 'basrpt_cost':
            costs = [self.calc_basrpt_cost(f, kwargs['V'], kwargs['N']) for f in contending_flows_list]
        elif cost_metric == 'fct':
            costs = []
            for f in contending_flows_list:
                cost, _ = self.estimate_time_to_completion(f)
                costs.append(cost)
        idx_min_cost = costs.index(min(costs))
        contending_flow = contending_flows_list[idx_min_cost]
        contending_flow_cost = costs[idx_min_cost]

        return contending_flow, contending_flow_cost, contending_flows_list, packets_scheduled_if_drop_all_contending_flows





































