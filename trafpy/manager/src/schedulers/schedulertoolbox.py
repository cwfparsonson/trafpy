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
    pass


class SchedulerToolbox_v2:
    def __init__(self, 
                 Graph, 
                 RWA, 
                 slot_size, 
                 packet_size=300, 
                 time_multiplexing=True, 
                 debug_mode=False):
        self.network = Graph 
        self.rwa = RWA
        self.slot_size = slot_size
        self.packet_size = packet_size
        self.time_multiplexing = time_multiplexing
        self.debug_mode = debug_mode
        
        self.reset()

    def reset(self):
        self.network = copy.deepcopy(self.network)

    def update_network_state(self, 
                             observation, 
                             hide_child_dependency_flows=True, 
                             reset_channel_capacities=True):
        '''
        If hide_child_dependency_flows is True, will only update scheduler network
        to see flows that are ready to be scheduled i.e. all parent flow dependencies
        have been completed. This is used for 'job-agnostic' scheduling systems
        which, rather than considering the job that each flow is part of, only consider
        the flow.
        
        If False, will just update network with all flows (even those that cannot yet
        be scheduled). This is used for 'job- & network- aware' scheduling systems.
        '''
        if type(observation) is dict:
            # network contained within observation dictionary
            self.network = copy.deepcopy(observation['network'])
        else:
            # assume observation has been given as network object
            self.network = copy.deepcopy(observation)

        if reset_channel_capacities:
            self.network = self.reset_channel_capacities_of_edges()
        if hide_child_dependency_flows:
            self.network = self.filter_unavailable_flows()


    def reset_channel_capacities_of_edges(self):
        '''Takes edges and resets their available capacities back to their maximum capacities.'''
        net = self.network
        for edge in net.edges:
            for channel in self.rwa.channel_names:
                # reset channel capacities of both ports
                net[edge[0]][edge[1]]['{}_to_{}_port'.format(edge[0], edge[1])]['channels'][channel] = net[edge[0]][edge[1]]['{}_to_{}_port'.format(edge[0], edge[1])]['max_channel_capacity']
                net[edge[1]][edge[0]]['{}_to_{}_port'.format(edge[1], edge[0])]['channels'][channel] = net[edge[0]][edge[1]]['{}_to_{}_port'.format(edge[1], edge[0])]['max_channel_capacity']
        # update global graph property
        net.graph['curr_nw_capacity_used'] = 0

        return net

    def filter_unavailable_flows(self):
        '''
        Takes a network and filters out any flow that is not ready to be scheduled
        yet i.e. has incomplete parent flow dependencies. Use this method to get
        network representation for 'job-agnostic' flow scheduling systems.
        '''
        net = copy.deepcopy(self.network)
        eps = net.graph['endpoints']
        for ep in eps:
            ep_queues = copy.deepcopy(net.nodes[ep]) # must make copy or will miss flows since length of list changes during iteration
            for ep_queue in ep_queues.values():
                for flow_dict in ep_queue['queued_flows']:
                    if flow_dict['can_schedule'] == 0:
                        # can't schedule, filter out of network
                        net = self.remove_flow_from_queue(flow_dict, net)
                    else:
                        # can schedule
                        pass
        
        # check no bad flows left in queue
        for ep in eps:
            ep_queues = net.nodes[ep]
            for ep_queue in ep_queues.values():
                for flow_dict in ep_queue['queued_flows']:
                    if flow_dict['can_schedule'] == 0:
                        raise Exception('Illegal flow(s) still present')
                    else:
                        pass

        return net

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

    def find_flow_idx(self, flow, flows):
        '''
        Finds flow idx in a list of flows. Assumes the following 
        flow features are unique and unchanged properties of each flow:
        - flow size
        - source
        - destination
        - time arrived
        - flow_id
        - job_id

        Args:
        - flow (dict): flow dictionary
        - flows (list of dicts) list of flows in which to find flow idx
        '''
        size = flow['size']
        src = flow['src']
        dst = flow['dst']
        time_arrived = flow['time_arrived']
        flow_id = flow['flow_id']
        job_id = flow['job_id']

        idx = 0
        for f in flows:
            if f['size']==size and f['src']==src and f['dst']==dst and f['time_arrived']==time_arrived and f['flow_id'] == flow_id and f['job_id'] == job_id:
                # flow found in flows
                return idx
            else:
                # flow not found, move to next f in flows
                idx += 1
        
        raise Exception('Flow not found in list of flows')


    def init_paths_and_packets(self, flow_dict):
        if flow_dict['k_shortest_paths'] is None:
            k_shortest_paths = self.rwa.k_shortest_paths(self.network,
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

    def gen_flow_packets(self, flow_size):
        num_packets = math.ceil(flow_size/self.packet_size) # round up 

        return self.packet_size, num_packets

    def collect_flow_info_dicts(self, 
                                path_channel_assignment_strategy='random', 
                                cost_function=None):
        '''
        Goes through network and collects useful dictionaries for making scheduling
        decisions.

        Args:
            path_channel_assignment_strategy (str): If 'random', allocates flows a randomly chosen
                path and channel. If None, does not allocate any path or channel; just
                uses whichever path and channel has already been assigned (if nothing
                assigned, will get an error later). If 'fair_share_num_flows', 
                distributes number of flows across channels and paths equally. If
                'fair_share_num_packets', distributes number of flow packets
                across channels and paths equally.
            cost_function (function): If not None, uses cost_function to assign a
                cost to each flow and stores this in a dictionary. cost_function
                must take a single flow dictionary argument.

        Returns:
            queued_flows (dict): Maps flow_id to corresponding flow dictionary.
            requested_edges (dict): Maps links (edges) in network being requested
                to corresponding flow ids requesting them.
            edge_to_flow_ids (dict): Maps edge to the list of flow ids requesting
                it.
            flow_id_to_cost (dict): Maps flow_id to corresponding cost of flow.

        '''
        queued_flows = {} 
        requested_edges = {}
        flow_id_to_cost = {} 

        if path_channel_assignment_strategy == 'fair_share_num_flows':
            channel_to_num_flows = {channel: 0 for channel in self.rwa.channel_names}
            edge_to_num_flows = {}
            for edge in self.network.edges:
                edge_to_num_flows[json.dumps(edge)] = 0
                edge_to_num_flows[json.dumps(edge[::-1])] = 0
        elif path_channel_assignment_strategy == 'fair_share_num_packets':
            channel_to_num_packets = {channel: 0 for channel in self.rwa.channel_names}
            edge_to_num_packets = {}
            for edge in self.network.edges:
                edge_to_num_packets[json.dumps(edge)] = 0
                edge_to_num_packets[json.dumps(edge[::-1])] = 0


        for ep in self.network.graph['endpoints']:
            queues = self.network.nodes[ep]
            for queue in queues.keys():
                for flow in queues[queue]['queued_flows']:
                    flow = self.init_paths_and_packets(flow)

                    if path_channel_assignment_strategy is None:
                        # do not assign any path or channel, use default
                        pass

                    elif path_channel_assignment_strategy == 'random':
                        # randomly choose a path and a channel
                        path_idx = np.random.choice(range(len(flow['k_shortest_paths'])))
                        flow['path'] = flow['k_shortest_paths'][path_idx]
                        flow['channel'] = np.random.choice(self.rwa.channel_names)

                    elif path_channel_assignment_strategy == 'fair_share_num_flows':
                        # distribute number of flows equally across paths and channels

                        # path
                        paths_num_flows = {idx: 0 for idx in range(len(flow['k_shortest_paths']))}
                        idx = 0
                        for path in flow['k_shortest_paths']:
                            for edge in self.get_path_edges(path):
                                paths_num_flows[idx] += edge_to_num_flows[json.dumps(edge)]
                            idx += 1
                        # choose path with lowest total number of flows on each edge
                        idx = min(paths_num_flows, key=paths_num_flows.get)
                        flow['path'] = flow['k_shortest_paths'][idx]
                        # register each edge in path as having another flow assigned
                        for edge in self.get_path_edges(flow['path']):
                            edge_to_num_flows[json.dumps(edge)] += 1

                        # channel
                        # choose channel with lowest total number of flows
                        flow['channel'] = min(channel_to_num_flows, key=channel_to_num_flows.get)
                        # register channel as having another flow assigned
                        channel_to_num_flows[flow['channel']] += 1

                    elif path_channel_assignment_strategy == 'fair_share_num_packets':
                        # distribute number of requesting packets equally across paths and channels

                        # path
                        paths_num_packets = {idx: 0 for idx in range(len(flow['k_shortest_paths']))}
                        idx = 0
                        for path in flow['k_shortest_paths']:
                            for edge in self.get_path_edges(path):
                                paths_num_packets[idx] += edge_to_num_packets[json.dumps(edge)]
                            idx += 1
                        # choose path with lowest total number of flows on each edge
                        idx = min(paths_num_packets, key=paths_num_packets.get)
                        flow['path'] = flow['k_shortest_paths'][idx]
                        # register each edge in path as having another flow assigned
                        for edge in self.get_path_edges(flow['path']):
                            edge_to_num_packets[json.dumps(edge)] += flow['packets']

                        # channel
                        # choose channel with lowest total number of flows
                        flow['channel'] = min(channel_to_num_packets, key=channel_to_num_packets.get)
                        # register channel as having another flow assigned
                        channel_to_num_packets[flow['channel']] += flow['packets']

                    else:
                        raise Exception('Unrecognised path_channel_assignment_strategy {}'.format(path_channel_assignment_strategy))

                    if 'unique_id' in flow:
                        identifier = 'unique_id'
                    else:
                        identifier = 'flow_id'

                    # assign a cost to this flow
                    if cost_function is None:
                        pass
                    else:
                        flow_id_to_cost[flow[identifier]] = cost_function(flow)

                    # collect flow
                    queued_flows[flow[identifier]] = flow

                    # collect requested edges
                    edges = self.get_path_edges(flow['path'])
                    for e in edges:
                        if json.dumps(e) in requested_edges.keys():
                            requested_edges[json.dumps(e)].append(flow[identifier]) # sort to keep order consistent
                        else:
                            requested_edges[json.dumps(e)] = [flow[identifier]] # sort to keep order consistent

        edge_to_flow_ids = {edge: [flow_id for flow_id in requested_edges[edge]] for edge in requested_edges.keys()}

        # edge_to_bandwidth = self.get_edge_to_maximum_bandwidth_dict(requested_edges)
        edge_to_bandwidth = self.get_edge_to_bandwidth_dict(requested_edges, max_bw=False)

        flow_info = {'queued_flows': queued_flows,
                       'requested_edges': requested_edges,
                       'edge_to_flow_ids': edge_to_flow_ids,
                       'edge_to_bandwidth': edge_to_bandwidth,
                       'flow_id_to_cost': flow_id_to_cost}

        return flow_info

    def get_edge_to_bandwidth_dict(self, requested_edges, max_bw=True):
        '''Goes through network and maps each edge to its maximum bandwidth.

        If max_bw, gets maximum possible bandwidth on each edge.
        If not max_bw, gets available bandwidth on each edge ASSUMES ONE CHANNEL.
        '''
        edge_to_bandwidth = {}
        for edge in requested_edges.keys():
            if max_bw:
                bandwidth = self.network[json.loads(edge)[0]][json.loads(edge)[1]]['{}_to_{}_port'.format(json.loads(edge)[0], json.loads(edge)[1])]['max_channel_capacity']
            else:
                # assume only one channel
                channel = self.rwa.channel_names[0]
                bandwidth = self.network[json.loads(edge)[0]][json.loads(edge)[1]]['{}_to_{}_port'.format(json.loads(edge)[0], json.loads(edge)[1])]['channels'][channel]
            edge_to_bandwidth[edge] = bandwidth

        return edge_to_bandwidth

    def allocate_available_bandwidth(self, 
                                     flow_info,
                                     resolution_strategy):
        '''
        Goes through each edge and allocates bandwidth available on that edge
        to requesting flows until either no bandwidth left to allocate or all
        requesting flows would be completed this time slot.

        If flow_id_to_cost is not None, will allocate bandwidth to flows in order of
        cost (prioritising low cost flows first). If flow_id_to_cost is None,
        must specify a valid resolution_strategy (e.g. 'random', 'fair_share', etc.).

        Args:
            flow_id_to_cost (dict): Dict mapping flow_id to corresponding flow cost.
            resolution_strategy (str): Which resolution strategy to use if flow_id_to_cost is None
                to allocate available bandwidht and resolve conflicts

        Returns:

        '''
        valid_resolution_strategies = ['cost', 'fair_share', 'random', 'first_fit']
        if resolution_strategy not in valid_resolution_strategies:
            raise Exception('resolution_strategy {} must be one of {}'.format(resolution_strategy, valid_resolution_strategies))

        edge_to_sorted_costs = {}
        edge_to_sorted_flow_ids = {}
        if resolution_strategy == 'cost':
            # sort requesting flow ids on each edge by order of cost (lowest to highest) -> is scheduling priority
            for edge in flow_info['edge_to_flow_ids'].keys():
                flow_ids = np.asarray([flow_id for flow_id in flow_info['edge_to_flow_ids'][edge]])
                costs = np.asarray([flow_info['flow_id_to_cost'][flow_id] for flow_id in flow_info['edge_to_flow_ids'][edge]])
                sorted_cost_index = np.argsort(costs)
                edge_to_sorted_costs[edge] = costs[sorted_cost_index]
                edge_to_sorted_flow_ids[edge] = flow_ids[sorted_cost_index]

        # init packets to schedule on each edge for each requesting flow
        edge_to_flow_id_to_packets_to_schedule = {edge:
                                                    {flow_id: 0 for flow_id in flow_info['edge_to_flow_ids'][edge]}
                                                  for edge in flow_info['edge_to_flow_ids'].keys()}

        # go through each edge and allocate available bandwidth
        for edge in flow_info['requested_edges'].keys():
            init_num_requests = len(flow_info['requested_edges'][edge])
            num_requests_left = len(flow_info['requested_edges'][edge])
            packets_scheduled_this_slot = 0

            # find max total packets can schedule this slot on this link
            max_info_per_slot = flow_info['edge_to_bandwidth'][edge] * self.slot_size # info transferred per slot == info transferred per unit time * number of time units (i.e. slot size)
            max_packets_per_slot = int(max_info_per_slot / self.packet_size) # round down

            # init packets left for flows requesting this edge
            flow_packets_left = {flow_id: flow_info['queued_flows'][flow_id]['packets'] for flow_id in flow_info['edge_to_flow_ids'][edge]}

            if resolution_strategy == 'cost':
                # iterate through flows requesting this edge in order of cost, prioritising lowest cost flows first
                sorted_flow_ids = iter(edge_to_sorted_flow_ids[edge])

            # choose flows to schedule for this edge in order of scheduling priority (cost)
            # until either cannot schedule any more flows this time slot or until
            # all flows requesting this edge will have been complete this time slot 
            while True:
                # new sub-slot of time <= time slot

                # find max packets can schedule for rest of time slot
                if resolution_strategy == 'fair_share':
                    max_packets_per_request = int((max_packets_per_slot-packets_scheduled_this_slot) / num_requests_left) # round down
                else:
                    max_packets_rest_of_time_slot = int(max_packets_per_slot-packets_scheduled_this_slot)

                
                ####################### COST-BASED SCHEDULING ######################
                if resolution_strategy == 'cost':
                    # select next highest priority flow
                    flow_id = next(sorted_flow_ids)
                    flow = flow_info['queued_flows'][flow_id]

                    # find number of packets to schedule for this highest priority flow
                    packets_to_schedule = min(flow['packets'], max_packets_rest_of_time_slot)

                    # update trackers to indicate this flow has been scheduled by corresponding number of packets
                    edge_to_flow_id_to_packets_to_schedule[edge][flow_id] += packets_to_schedule
                    flow_packets_left[flow_id] -= packets_to_schedule
                    packets_scheduled_this_slot += packets_to_schedule
                    if flow_packets_left[flow_id] == 0:
                        num_requests_left -= 1

                    if packets_scheduled_this_slot == max_packets_per_slot or num_requests_left == 0:
                        # finished scheduling time slot for this edge, move to next edge
                        break
                    else:
                        # move to next sub-slot of overall time slot for this edge
                        pass


                ############################# FAIR SHARE SCHEDULING ###########################
                elif resolution_strategy == 'fair_share':
                    # find smallest packets left of remaining requests on this edge
                    non_zero_packets = [packet for packet in flow_packets_left.values() if packet != 0]
                    smallest_packets_left = min(non_zero_packets)

                    # find packets to schedule per request for this sub slot
                    packets_per_request = min(smallest_packets_left, max_packets_per_request)

                    # fair share by scheduling packets per request equally for each request for this sub-slot
                    for flow_id in flow_packets_left.keys():
                        if flow_packets_left[flow_id] != 0:
                            edge_to_flow_id_to_packets_to_schedule[edge][flow_id] += packets_per_request
                            flow_packets_left[flow_id] -= packets_per_request
                            packets_scheduled_this_slot += packets_per_request
                            if flow_packets_left[flow_id] == 0:
                                # all packets of this flow have now been scheduled
                                num_requests_left -= 1
                    
                    if packets_scheduled_this_slot >= max_packets_per_slot-init_num_requests or num_requests_left == 0:
                        # finished fair sharing this slot for this edge
                        break
                    else:
                        # move to next sub-slot for this edge
                        pass

                ############################# RANDOM & FIRST FIT SCHEDULING ###########################
                elif resolution_strategy == 'random' or resolution_strategy == 'first_fit':
                    # randomly select a flow to schedule
                    flow_id = np.random.choice(list(flow_packets_left.keys()))
                    flow = flow_info['queued_flows'][flow_id]
                    
                    # find number of packets to schedule for this flow
                    packets_to_schedule = min(flow['packets'], max_packets_rest_of_time_slot)

                    # update trackers to indicate this flow has been scheduled by corresponding number of packets
                    edge_to_flow_id_to_packets_to_schedule[edge][flow_id] += packets_to_schedule
                    flow_packets_left[flow_id] -= packets_to_schedule
                    packets_scheduled_this_slot += packets_to_schedule
                    if flow_packets_left[flow_id] == 0:
                        num_requests_left -= 1

                    if packets_scheduled_this_slot == max_packets_per_slot or num_requests_left == 0:
                        # finished scheduling time slot for this edge, move to next edge
                        break
                    else:
                        # move to next sub-slot of overall time slot for this edge
                        pass

                else:
                    raise Exception('resolution_strategy {} does not seem to be implemented.'.format(resolution_strategy))



        # find which flows were chosen on each edge, and collect how many packets were scheduled for each chosen flow
        flow_id_to_packets_to_schedule_per_edge = {flow_id: [] for flow_id in flow_info['queued_flows'].keys()}
        for edge in edge_to_flow_id_to_packets_to_schedule.keys():
            for flow_id in edge_to_flow_id_to_packets_to_schedule[edge].keys():
                if edge_to_flow_id_to_packets_to_schedule[edge][flow_id] != 0:
                    flow_id_to_packets_to_schedule_per_edge[flow_id].append(edge_to_flow_id_to_packets_to_schedule[edge][flow_id])

        scheduling_info = {'edge_to_flow_id_to_packets_to_schedule': edge_to_flow_id_to_packets_to_schedule,
                           'flow_id_to_packets_to_schedule_per_edge': flow_id_to_packets_to_schedule_per_edge}
        cost_info = {'flow_id_to_cost': flow_info['flow_id_to_cost'],
                     'edge_to_sorted_costs': edge_to_sorted_costs,
                     'edge_to_sorted_flow_ids': edge_to_sorted_flow_ids}

        return scheduling_info, cost_info


    def resolve_contentions_and_set_up_flow(self,
                                            flow, 
                                            chosen_flows, 
                                            flow_info,
                                            scheduling_info,
                                            cost_info,
                                            resolution_strategy):
        '''
        If contention found, will resolve contention using resolution strategy.

        Cost resolution strategy -> choose flow with lowest cost.
        Random resolution strategy -> choose random flow.
        '''
        if 'unique_id' in flow:
            identifier = 'unique_id'
        else:
            identifier = 'flow_id'

        valid_resolution_strategies = ['cost', 'random', 'fair_share', 'first_fit']
        if resolution_strategy not in valid_resolution_strategies:
            raise Exception('resolution_strategy {} must be one of {}'.format(resolution_strategy, valid_resolution_strategies))

        chosen_flow_ids = {f[identifier]: None for f in chosen_flows}
        if self.debug_mode:
            print('\n-----')
            print('considering flow: {}'.format(flow))
            print('chosen flow ids: {}'.format(chosen_flow_ids))
        removed_flows = []
        loops = 0
        while True:
            if loops >= 1e6: # temporary to stop infinite loop bugs -> may need to remove if have v large no. flows and therefore require many loops
                raise Exception('Auto exited while loop. Delete line if confident no bugs, or increase number of loops if have very large number of flows to go through.')
            loops += 1
            if self.debug_mode:
                print('flows removed:\n{}'.format(removed_flows))
            if self.check_connection_valid(flow):
                # no contention(s) -> can set up flow
                if self.debug_mode:
                    print('no contention, can set up flow.')
                self.set_up_connection(flow)
                chosen_flows.append(flow)

                # if any left over bandwidth, try re-establish any removed flows but now with fewer packets
                # try establish in reverse order (since this will go from lowest cost flows to highest cost)
                for f in reversed(removed_flows):
                    if self.debug_mode:
                        print('checking if any leftover bandwidth for previously removed flow {}'.format(f))
                    # 1. check lowest available bandwidth remaining in flow path
                    lowest_edge_bandwidth = self.get_lowest_edge_bandwidth(path=f['path'], max_bw=False, channel=f['channel'])
                    max_info = lowest_edge_bandwidth * self.slot_size 
                    max_packets = int(max_info / self.packet_size) # round down

                    # 2. if lowest bw != 0, set removed flow packets with min(flow packets, max poss packets with bw)
                    if lowest_edge_bandwidth != 0:
                        packets_to_schedule = min(f['packets'], max_packets)
                        if self.debug_mode:
                            print('bandwidth available on flow\'s path and channel. Setting up removed flow to schedule {} of its packets.'.format(packets_to_schedule))

                        # update flow packets, set up connection, and append flow to chosen_flows
                        f['packets_this_slot'] = packets_to_schedule
                        self.set_up_connection(f)
                        chosen_flows.append(f)

                    else:
                        # no bandwidth left, cannot establish this removed flow
                        if self.debug_mode:
                            print('no bandwidth available on flow\' path and channel. Cannot set up again.')
                        pass

                return chosen_flows

            else:
                # there's a conflict with an already chosen flow
                # find contending flow -> see which flow to establish
                if self.debug_mode:
                    print('conflict detected')
                flow_edges = self.get_path_edges(flow['path'])
                bandwidth_requested = (flow['packets_this_slot'] * flow['packet_size'])/self.slot_size
                if self.debug_mode:
                    print('flow {} bandwidth requested: {}'.format(flow[identifier], bandwidth_requested))
                for edge in flow_edges:
                    if not self.check_edge_valid(flow, edge):
                        # contention is on this edge
                        if self.debug_mode:
                            print('found contention on edge {}'.format(edge))


                        ######################### FIRST FIT RESOLUTION #########################
                        if resolution_strategy == 'first_fit':
                            # schedule first flow which fits bandwidth, therefore do not attempt to resolve any conflicts which arise after a flow has already been schedueld

                            # if any bandwidth available, try fill with this flow
                            # 1. check lowest available bandwidth remaining in flow path
                            lowest_edge_bandwidth = self.get_lowest_edge_bandwidth(path=flow['path'], max_bw=False, channel=flow['channel'])
                            max_info = lowest_edge_bandwidth * self.slot_size
                            max_packets = int(max_info / self.packet_size) # round down
                            # 2. if lowest bw != 0, set removed flow packets with min(flow packets, max poss packets with bw)
                            if lowest_edge_bandwidth != 0:
                                packets_to_schedule = min(flow['packets'], max_packets)
                                flow['packets_this_slot'] = packets_to_schedule
                                self.set_up_connection(flow)
                                chosen_flows.append(flow)
                            else:
                                # no bandwidth left, cannot establish this flow
                                pass

                            # # if any extra bandwidth available for any chosen flows, fill
                            # for f in chosen_flows:
                                # # 1. check lowest available bandwidth remaining in flow path
                                # lowest_edge_bandwidth = self.get_lowest_edge_bandwidth(path=f['path'], max_bw=False, channel=f['channel'])
                                # max_info = lowest_edge_bandwidth * self.slot_size 
                                # max_packets = int(max_info / self.packet_size) # round down

                                # # 2. if lowest bw != 0, set removed flow packets with min(flow packets, max poss packets with bw)
                                # if lowest_edge_bandwidth != 0:
                                    # # temporarily remove so can update flow packets this slot
                                    # self.take_down_connection(f)
                                    # chosen_flows.remove(f)
                                    # packets_to_schedule = min(f['packets'], max_packets)

                                    # # update flow packets, set up connection, and re-append flow to chosen_flows
                                    # f['packets_this_slot'] = packets_to_schedule
                                    # self.set_up_connection(f)
                                    # chosen_flows.append(f)
                                # else:
                                    # # no bandwidth left, cannot establish this removed flow
                                    # pass

                            return chosen_flows

                        ######################### RANDOM RESOLUTION #########################
                        elif resolution_strategy == 'random':
                            # randomly decide if will establish flow
                            establish = np.random.choice([True, False])
                            if not establish:
                                # do not establish flow, re-establish any flows that were taken down
                                for f in removed_flows:
                                    self.set_up_connection(f)
                                    chosen_flows.append(f)
                                return chosen_flows
                            else:
                                # remove conflicting flow -> move to next while loop to try again to re-establish flow
                                flow_ids = list(scheduling_info['edge_to_flow_id_to_packets_to_schedule'][json.dumps(edge)].keys())
                                for _id in flow_ids:
                                    if _id in chosen_flow_ids:
                                        # found flow to discard
                                        found_f = False
                                        i = 0
                                        while not found_f:
                                            f = chosen_flows[i]
                                            if f[identifier] == _id:
                                                if self.debug_mode:
                                                    print('found established flow, take down')
                                                self.take_down_connection(f)
                                                chosen_flows.remove(f)
                                                del chosen_flow_ids[f[identifier]]
                                                removed_flows.append(f)
                                                found_f = True
                                                if self.debug_mode:
                                                    print('moving to next while loop iter to try set up flow again...')
                                            else:
                                                i += 1
                                        # found flow in chosen flows, iterating through chosen flow ids
                                        break


                        ######################### COST RESOLUTION #########################
                        elif resolution_strategy == 'cost':
                            # if flow has lower cost than one of the contentions, remove highest cost (least contentious) flow and try to set up connection again
                            costs = cost_info['edge_to_sorted_costs'][json.dumps(edge)]
                            flow_ids = cost_info['edge_to_sorted_flow_ids'][json.dumps(edge)]
                            if self.debug_mode:
                                print('flow ids requesting this edge: {}'.format(flow_ids))
                                print('costs: {}'.format(costs))
                            for idx in reversed(range(len(flow_ids))):
                                _id = flow_ids[idx]
                                _cost = costs[idx]
                                if _id in chosen_flow_ids:
                                    # this flow has previously been chosen, check if should keep or discard
                                    if _cost < cost_info['flow_id_to_cost'][flow[identifier]]:
                                        if self.debug_mode:
                                            print('cost of prospective flow greater than already established flow, do not set up')
                                        # already established flow has lower cost -> do not establish flow, re-establish any flows that were taken down
                                        for f in removed_flows:
                                            self.set_up_connection(f)
                                            chosen_flows.append(f)
                                        return chosen_flows 
                                    else:
                                        if self.debug_mode:
                                            print('cost of prospective flow less than established flow, try to establish')
                                        # remove higher cost flow -> move to next while loop to try again to re-establish flow
                                        found_f = False
                                        i = 0
                                        while not found_f:
                                            f = chosen_flows[i]
                                            if f[identifier] == _id:
                                                if self.debug_mode:
                                                    print('found high cost established flow {}, take down'.format(_id))
                                                self.take_down_connection(f)
                                                chosen_flows.remove(f)
                                                del chosen_flow_ids[f[identifier]]
                                                removed_flows.append(f)
                                                found_f = True
                                                if self.debug_mode:
                                                    print('moving to next while loop iter to try set up flow again...')
                                            else:
                                                i += 1
                                        # found flow in chosen flows, iterating through chosen flow ids
                                        break







                            else:
                                continue 
                    else:
                        continue


    def get_lowest_edge_bandwidth(self, path, max_bw=True, channel=None):
        '''Goes through path edges and finds bandwidth of lowest bandwidth edge port.

        If max_bw, will return maximum possible bandwith of lowest max bandwidth edge.
        If not, will return available bandwidth of lowest available bandwidth edge.

        N.B. if not max_bw, MUST given channel (since checking bandwidth available on
        edge)
        '''
        if not max_bw and channel is None:
            raise Exception('If not max_bw, must specify channel to check available bandwidth on channel for each edge in path.')

        lowest_edge_bw = float('inf')
        edges = self.get_path_edges(path)
        for edge in edges:
            if max_bw:
                # find maximum bandwidth of edge with lowest maximum bandwidth
                bw = self.network[edge[0]][edge[1]]['{}_to_{}_port'.format(edge[0], edge[1])]['max_channel_capacity']
            else:
                # find maximum bandwidth of edge with lowest maximum bandwidth
                bw = self.network[edge[0]][edge[1]]['{}_to_{}_port'.format(edge[0], edge[1])]['channels'][channel]
            if bw < lowest_edge_bw:
                lowest_edge_bw = bw 

        return lowest_edge_bw
            






    def check_connection_valid(self, flow, num_decimals=6):
        '''
        Returns False if setting up connection would result in -ve 
        bandwidth on at least one link in network.
        '''
        edges = self.get_path_edges(flow['path'])

        num_edges = len(edges)
        for edge in range(num_edges):
            node_pair = edges[edge]

            # check that establishing this flow is valid given edge capacity constraints
            if not self.check_edge_valid(flow, node_pair, num_decimals):
                # DEBUG
                if self.debug_mode:
                    for edge in self.network.edges:
                        for channel in self.rwa.channel_names:
                            # src-dst
                            bw = self.get_channel_bandwidth(edge, channel)
                            print('edge: {} | channel: {} | bandwidth remaining: {}'.format(edge, channel, bw))
                            # dst-src
                            edge = edge[::-1]
                            bw = self.get_channel_bandwidth(edge, channel)
                            print('edge: {} | channel: {} | bandwidth remaining: {}'.format(edge, channel, bw))

                return False

        return True

    def check_edge_valid(self, flow, edge, num_decimals=6):
        info_to_transfer_this_slot = flow['packets_this_slot'] * flow['packet_size']
        capacity_used_this_slot = round(info_to_transfer_this_slot / self.slot_size, num_decimals) # info units of this flow transferred this time slot == capacity used on each channel in flow's path this time slot
        # if self.network[edge[0]][edge[1]]['channels'][flow['channel']] - capacity_used_this_slot < 0:
        if self.network[edge[0]][edge[1]]['{}_to_{}_port'.format(edge[0], edge[1])]['channels'][flow['channel']] - capacity_used_this_slot < 0:
            return False
        else:
            return True

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
        return [path[edge:edge+2] for edge in range(num_edges)]


    def set_up_connection(self, flow, num_decimals=6):
        '''
        Sets up connection between src-dst node pair by removing capacity from
        all edges in path connecting them. Also updates graph's global curr 
        network capacity used property
        
        Args:
        - flow (dict): flow dict containing flow info to set up
        '''
        if flow['can_schedule'] == 0:
            raise Exception('Tried to set up flow {}, but this flow cannot yet be scheduled (can_schedule == 0)! Scheduler should not be giving invalid chosen flow sets to the environment.'.format(flow)) 

        # if self.debug_mode:
            # print('Setting up connection for flow {}'.format(flow))

        if not self.check_connection_valid(flow):
            raise Exception('Tried to set up connection for flow {} but would result in -ve bandwidth on at least one edge in network.'.format(flow))

        path = flow['path']
        channel = flow['channel']
        packet_size = flow['packet_size']
        packets_this_slot = flow['packets_this_slot']

        info_to_transfer_this_slot = packets_this_slot * packet_size
        capacity_used_this_slot = round(info_to_transfer_this_slot / self.slot_size, num_decimals) # info units of this flow transferred this time slot == capacity used on each channel in flow's path this time slot

        edges = self.get_path_edges(path)

        num_edges = len(edges)
        for edge in range(num_edges):
            node_pair = edges[edge]

            # update edge capacity remaining after establish this flow
            init_bw = self.get_channel_bandwidth(node_pair, channel)
            # self.network[node_pair[0]][node_pair[1]]['channels'][channel] -= capacity_used_this_slot
            # self.network[node_pair[0]][node_pair[1]]['channels'][channel] = round(self.network[node_pair[0]][node_pair[1]]['channels'][channel], num_decimals)
            self.network[node_pair[0]][node_pair[1]]['{}_to_{}_port'.format(node_pair[0], node_pair[1])]['channels'][channel] -= capacity_used_this_slot
            self.network[node_pair[0]][node_pair[1]]['{}_to_{}_port'.format(node_pair[0], node_pair[1])]['channels'][channel] = round(self.network[node_pair[0]][node_pair[1]]['{}_to_{}_port'.format(node_pair[0], node_pair[1])]['channels'][channel], num_decimals)
            # if self.debug_mode:
                # print('Updated {} capacity for {}: {} -> {}'.format(channel, node_pair, init_bw, self.get_channel_bandwidth(node_pair, channel)))


            # update global graph property
            self.network.graph['curr_nw_capacity_used'] += capacity_used_this_slot
        self.network.graph['num_active_connections'] += 1

    def take_down_connection(self, flow, num_decimals=6):
        '''
        Removes established connection by adding capacity back onto all edges
        in the path connecting the src-dst node pair. Also updates graph's
        global curr network capacity used property

        Args:
        - flow (dict): flow dict containing info of flow to take down
        '''
        # if self.debug_mode:
            # print('Taking down connection for flow {}'.format(flow))
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
            self.network[node_pair[0]][node_pair[1]]['{}_to_{}_port'.format(node_pair[0], node_pair[1])]['channels'][channel] += capacity_used_this_slot
            self.network[node_pair[0]][node_pair[1]]['{}_to_{}_port'.format(node_pair[0], node_pair[1])]['channels'][channel] = round(self.network[node_pair[0]][node_pair[1]]['{}_to_{}_port'.format(node_pair[0], node_pair[1])]['channels'][channel], num_decimals)
            # if self.debug_mode:
                # print('Updated {} capacity for {}: {} -> {}'.format(channel, node_pair, init_bw, self.get_channel_bandwidth(node_pair, channel)))
            # update global graph property
            self.network.graph['curr_nw_capacity_used'] -= capacity_used_this_slot
        self.network.graph['num_active_connections'] -= 1


    def get_channel_bandwidth(self, edge, channel):
        '''Gets current channel bandwidth left on a given edge in the network.'''
        return self.network[edge[0]][edge[1]]['{}_to_{}_port'.format(edge[0], edge[1])]['channels'][channel]
        # try:
            # return self.network[edge[0]][edge[1]]['{}_to_{}_port'.format(edge[0], edge[1])]['channels'][channel]
        # except KeyError:
            # return self.network[edge[1]][edge[0]]['{}_to_{}_port'.format(edge[0], edge[1])]['channels'][channel]

    def find_flow_queue(self, flow):
        '''
        Finds queue of flow in network
        '''
        sn = flow['src']
        dn = flow['dst']
        flow_queue = self.network.nodes[sn][dn]['queued_flows']

        return flow_queue

    def estimate_time_to_completion(self, flow):
        path_links = self.get_path_edges(flow['path'])
        link_bws = []
        for link in path_links:
            link_bws.append(self.network[link[0]][link[1]]['{}_to_{}_port'.format(link[0], link[1])]['max_channel_capacity'])
        lowest_bw = min(link_bws)
        
        size_per_slot = lowest_bw/(1/self.slot_size)
        packets_per_slot = int(size_per_slot / flow['packet_size']) # round down 
        if packets_per_slot == 0:
            raise Exception('Encountered 0 packets that can be transferred per time slot. Either decrease packet size or increase time slot size.')
        slots_to_completion = math.ceil(flow['packets']/packets_per_slot) # round up
        completion_time = slots_to_completion * self.slot_size

        return completion_time






