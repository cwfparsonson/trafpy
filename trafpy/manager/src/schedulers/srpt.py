from trafpy.manager.src.schedulers.schedulertoolbox import SchedulerToolbox

import numpy as np
import networkx as nx
import copy
import math
import sys
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import itertools
import time
import pandas as pd
from tabulate import tabulate
import json


class SRPT(SchedulerToolbox):

    def __init__(self, Graph, RWA, slot_size, packet_size=300, time_multiplexing=True, debug_mode=False, scheduler_name='srpt'):
        super().__init__(Graph, RWA, slot_size, packet_size, time_multiplexing, debug_mode)
        self.scheduler_name = scheduler_name

    def get_scheduler_action(self, observation):
        '''
        Uses observation and chosen rwa action(s) to construct schedule for this
        timeslot
        '''
        # update scheduler network & new flow states
        self.update_network_state(observation, hide_child_dependency_flows=True)

        # choose which flows to schedule for this time slot
        chosen_flows = []
        for ep in self.SchedulerNetwork.graph['endpoints']:
            queues = self.SchedulerNetwork.nodes[ep]
            for queue in queues.keys():
                queued_flows = queues[queue]['queued_flows']
                completion_times = queues[queue]['completion_times']
                num_queued_flows = len(queues[queue]['queued_flows'])
                if num_queued_flows == 0:
                    # no flows queued, move to next queue
                    continue
                else:
                    # queued flows present
                    chosen_flow, _ = self.find_shortest_flow_in_queue(queued_flows,completion_times)
                    if self.debug_mode:
                        print('\nAttempting to establish flow {}'.format(chosen_flow))
                   
                    # check for contentions
                    contending_flows = [None]
                    contending_flow = None
                    establish_flow = False
                    if len(chosen_flows) != 0:
                        establish_flow, p, c, packets_this_slot = self.look_for_available_lightpath(chosen_flow,chosen_flows)
                        chosen_flow['path'], chosen_flow['channel'], chosen_flow['packets_this_slot'] = p, c, packets_this_slot
                        if not establish_flow:
                            contending_flows,contending_flow,contending_flow_fct,p,c,packets_this_slot = self.find_contending_flow(chosen_flow,chosen_flows)
                            chosen_flow['path'], chosen_flow['channel'], chosen_flow['packets_this_slot'] = p, c, packets_this_slot
                            comp_time, _ = self.estimate_time_to_completion(chosen_flow)

                            if contending_flow_fct > comp_time:
                                # new choice has lower fct that established flow
                                establish_flow = True
                            else:
                                # established flow has lower fct, do not choose
                                pass
                        else:
                            # rwa was completed
                            pass
                    else:
                        # no existing chosen flows yet, can choose flow
                        establish_flow = True
                    
                    if establish_flow:
                        if self.debug_mode:
                            print('Chosen flow {} can be established'.format(chosen_flow))
                        for contending_flow in contending_flows:
                            try:
                                chosen_flows.remove(contending_flow)
                                self.take_down_connection(contending_flow)
                            except (NameError, ValueError):
                                # already not present
                                pass
                        chosen_flows.append(chosen_flow)
                        self.set_up_connection(chosen_flow)
                    else:
                        if self.debug_mode:
                            print('Chosen flow could not be established')
                        # contention was found and lost
                        pass


        if self.debug_mode:
            print('Channel capacity status after finished choosing actions:')
            net = self.SchedulerNetwork
            for edge in net.edges:
                for channel in net[edge[0]][edge[1]]['channels']:
                    # reset channel capacity
                    chan_cap_available = self.get_channel_bandwidth(edge, channel)
                    chan_cap_max = net[edge[0]][edge[1]]['max_channel_capacity']
                    print('Available {} capacity for {}: {} / {}'.format(channel, edge, chan_cap_available, chan_cap_max))

                            

        return chosen_flows

    def get_action(self, observation, print_processing_time=False):

        # scheduler action
        self.time_get_action_start = time.time()
        chosen_flows = self.get_scheduler_action(observation)
        action = {'chosen_flows': chosen_flows}
        self.time_get_action_end = time.time()

        if print_processing_time:
            self.display_get_action_processing_time()

        return action

    def display_get_action_processing_time(self, num_decimals=8):
        get_action_time = self.time_get_action_end - self.time_get_action_start

        # create table
        summary_dict = {'Get Action': [round(get_action_time, num_decimals)]}
        df = pd.DataFrame(summary_dict)
        print('')
        print(tabulate(df, showindex=False, headers='keys', tablefmt='psql'))

    
    def find_contending_flow(self, chosen_flow, chosen_flows):
        '''
        Goes through chosen flow possible path & channel combinations & 
        compares to path-channel combinations in chosen flows. Saves all
        contentions that arise. When all possible contentions have been checked,
        finds the 'most contentious' (i.e. shortest flow completion time) in
        chosen_flows and returns this as the contending flow (since other
        flows in contending_flows will have a higher FCT than this most contentious flow
        and therefore if the chosen flow has a lower FCT than the most contentious flow,
        it will also have a lower FCT than all competing flows and therefore should
        replace all contending flows)
        '''
        contending_flows = self.find_all_contending_flows(chosen_flow, chosen_flows, cost_metric='fct')

        chosen_path, chosen_channel, contending_flow, contending_flow_cost, contending_flows_list = self.choose_channel_and_path_using_contending_flows(contending_flows)

        # find number of packets that would be able to schedule for chosen flow if successful

        # can schedule up to the number of packets removed by removing contending flows
        # OR up to lowest non-0 bandwidth edge on chosen flow's path (whichever is lowest
        # is the limiting factor determining the number of packets which can be scheduled)

        # 1. if all edges were empty, what would be the maximum possible number of packets that could transfer this time slot?
        max_packets_available_if_all_edges_empty = self.get_maximum_packets_available_if_all_edges_empty(chosen_flow, chosen_path)

        # 2. how many packets can be transferred on chosen path edges which are not in the contending path edges (and therefore will not change even if contending flows are dropped)?
        packets_available_outside_contending_edges = self.get_packets_available_outside_contending_edges(chosen_flow, chosen_path, chosen_channel, contending_flows_list)

        # 3. if all contending flows were dropped, how many packets would be made AVAILABLE to be scheduled for the chosen flow?
        packets_available_if_drop_all_contending_flows = self.get_packets_available_if_drop_all_contending_flows(chosen_flow, chosen_path, chosen_channel, contending_flows_list)

        # 4. given the constraints of 1. the flow's remaining packets and 2. the bandwidth constraints of edges outside the contending edges, what is the maximum number of packets that the flow could schedule this time slot?
        max_packets_requested_by_chosen_flow = self.get_maximum_packets_requestable_by_flow(chosen_flow, max_packets_available_if_all_edges_empty, packets_available_outside_contending_edges)

        # 5. if drop all contending flows and schedule this chosen flow, how many packets would actually end up being SCHEDULED for this flow?
        packets_scheduled_if_drop_all_contending_flows = min(max_packets_available_if_all_edges_empty, packets_available_if_drop_all_contending_flows)

        # print('Max packets requested: {} | Packets that would be scheduled if drop all contending flows: {}'.format(max_packets_requested_by_chosen_flow, packets_scheduled_if_drop_all_contending_flows))
        # to make scheduler maximise its utility of available channel bandwidth, make sure
        # that would only drop the minimum number of necessery contending flows to allow the chosen flow to 
        # be scheduled
        if max_packets_requested_by_chosen_flow < packets_scheduled_if_drop_all_contending_flows:
            contending_flow, contending_flow_cost, contending_flows_list, packets_scheduled_if_drop_all_contending_flows = self.select_minimum_number_of_contending_flows_to_drop(chosen_flow, chosen_path, chosen_channel, contending_flows_list, max_packets_requested_by_chosen_flow, max_packets_available_if_all_edges_empty, 'fct')

        num_packets_this_slot = min(max_packets_requested_by_chosen_flow, packets_scheduled_if_drop_all_contending_flows)
         
        return contending_flows_list, contending_flow, contending_flow_cost, chosen_path, chosen_channel, num_packets_this_slot 

        




class SRPT_v2:
    def __init__(self,
                 Graph,
                 RWA,
                 slot_size,
                 packet_size=300,
                 time_multiplexing=True,
                 debug_mode=False,
                 scheduler_name='srpt_v2'):
        self.scheduler = SchedulerToolbox(Graph, RWA, slot_size, time_multiplexing, debug_mode)
        self.scheduler_name = scheduler_name

    def get_action(self, observation, print_processing_time=False):
        chosen_flows = self.get_scheduler_action(observation)
        action = {'chosen_flows': chosen_flows}

        return action

    def get_scheduler_action(self, observation):
        # print('\n\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> get scheduler action')
        # update scheduler network & new flow states
        self.scheduler.update_network_state(observation, hide_child_dependency_flows=True)

        chosen_flows = []

        # Collect dicts of i) all flows in queues and ii) which links (edges) each flow is requesting
        queued_flows = {} # maps flow_id to corresponding flow dict
        requested_edges = {} # maps links (edges) being requested to corresponding flow_ids
        flow_id_to_cost = {} # maps flow_id to corresponding cost
        for ep in self.scheduler.SchedulerNetwork.graph['endpoints']:
            queues = self.scheduler.SchedulerNetwork.nodes[ep]
            for queue in queues.keys():
                for flow in queues[queue]['queued_flows']:
                    flow = self.scheduler.init_paths_and_packets(flow)

                    # randomly choose a path and a channel
                    path_idx = np.random.choice(range(len(flow['k_shortest_paths'])))
                    flow['path'] = flow['k_shortest_paths'][path_idx]
                    flow['channel'] = np.random.choice(self.scheduler.RWA.channel_names)

                    # assign a cost to this flow
                    fct, _ = self.scheduler.estimate_time_to_completion(flow)
                    flow_id_to_cost[flow['flow_id']] = fct

                    # collect flow
                    queued_flows[flow['flow_id']] = flow

                    # collect requested edges
                    edges = self.scheduler.get_path_edges(flow['path'])
                    for e in edges:
                        if json.dumps(sorted(e)) in requested_edges.keys():
                            requested_edges[json.dumps(sorted(e))].append(flow['flow_id']) # sort to keep order consistent
                        else:
                            requested_edges[json.dumps(sorted(e))] = [flow['flow_id']] # sort to keep order consistent

        # collect which flows are requesting each edge
        edge_to_flow_ids = {edge: [flow_id for flow_id in requested_edges[edge]] for edge in requested_edges.keys()}

        # get maximum capacity on each edge being requested
        edge_to_bandwidth = {}
        for edge in requested_edges.keys():
            try:
                bandwidth = self.scheduler.SchedulerNetwork[json.loads(edge)[0]][json.loads(edge)[1]]['max_channel_capacity']
            except KeyError:
                bandwidth = self.scheduler.SchedulerNetwork[json.loads(edge)[1]][json.loads(edge)[0]]['max_channel_capacity']
            edge_to_bandwidth[edge] = bandwidth

        # sort requesting flow ids on each edge by order of cost (lowest to highest) -> is scheduling priority
        edge_to_sorted_costs = {}
        edge_to_sorted_flow_ids = {}
        for edge in edge_to_flow_ids.keys():
            flow_ids = np.asarray([flow_id for flow_id in edge_to_flow_ids[edge]])
            costs = np.asarray([flow_id_to_cost[flow_id] for flow_id in edge_to_flow_ids[edge]])
            sorted_cost_index = np.argsort(costs)
            edge_to_sorted_costs[edge] = costs[sorted_cost_index]
            edge_to_sorted_flow_ids[edge] = flow_ids[sorted_cost_index]

        # init packets to schedule on each edge for each requesting flow
        edge_to_flow_id_to_packets_to_schedule = {edge:
                                                    {flow_id: 0 for flow_id in edge_to_flow_ids[edge]}
                                                  for edge in edge_to_flow_ids.keys()}

        # go through each edge and allocate available bandwidth in order of cost
        for edge in requested_edges.keys():
            init_num_requests = len(requested_edges[edge])
            num_requests_left = len(requested_edges[edge])
            packets_scheduled_this_slot = 0

            # find max total packets can schedule this slot on this link
            max_info_per_slot = edge_to_bandwidth[edge] * self.scheduler.slot_size # info transferred per slot == info transferred per unit time * number of time units (i.e. slot size)
            max_packets_per_slot = int(max_info_per_slot / self.scheduler.packet_size) # round down

            # init packets left for flows requesting this edge
            flow_packets_left = {flow_id: queued_flows[flow_id]['packets'] for flow_id in edge_to_flow_ids[edge]}

            # choose flows to schedule for each edge in order of scheduling priority (cost)
            sorted_flow_ids = iter(edge_to_sorted_flow_ids[edge])
            while True:
                # new sub-slot

                # find max packets can schedule for rest of time slot
                max_packets_rest_of_time_slot = int(max_packets_per_slot-packets_scheduled_this_slot)

                # select next highest priority flow
                flow_id = next(sorted_flow_ids)
                flow = queued_flows[flow_id]

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

        # find which flows were chosen on each edge, and collect how many packets were scheduled for each chosen flow
        flow_id_to_packets_to_schedule_per_edge = {flow_id: [] for flow_id in queued_flows.keys()}
        for edge in edge_to_flow_id_to_packets_to_schedule.keys():
            for flow_id in edge_to_flow_id_to_packets_to_schedule[edge].keys():
                if edge_to_flow_id_to_packets_to_schedule[edge][flow_id] != 0:
                    flow_id_to_packets_to_schedule_per_edge[flow_id].append(edge_to_flow_id_to_packets_to_schedule[edge][flow_id])

        # collect chosen flows and corresponding packets to schedule for the chosen flows
        for flow_id in queued_flows.keys():
            if flow_id not in flow_id_to_packets_to_schedule_per_edge or len(flow_id_to_packets_to_schedule_per_edge[flow_id]) == 0:
                # flow was not chosen to be scheduled on any edge
                pass
            else:
                # flow was chosen to be scheduled on an edge
                flow = queued_flows[flow_id]

                # packets to schedule for each flow limited by the edge in their path with lowest number of schedulable packets
                flow['packets_this_slot'] = min(flow_id_to_packets_to_schedule_per_edge[flow_id])

                # check that flow was also selected to be scheduled on a bandwidth-limited end point link (if it wasn't, cannot schedule this flow)
                info_to_transfer_this_slot = flow['packets_this_slot'] * flow['packet_size']
                bandwidth_requested = info_to_transfer_this_slot / self.scheduler.slot_size # info units of this flow transferred this time slot == capacity used on each channel in flow's path this time slot
                if bandwidth_requested > self.scheduler.SchedulerNetwork.graph['ep_link_capacity']:
                    # flow must only have been selected for high capacity non-endpoint links and not been given any end point link capacity, do not schedule flow
                    pass
                else:
                    # flow must have been allocated bandwidth on at least one end point link, check for contentions and try to establish flow 
                    chosen_flows = self.resolve_contentions_and_set_up(flow, chosen_flows, requested_edges, flow_id_to_cost, edge_to_sorted_costs, edge_to_sorted_flow_ids, flow_id_to_packets_to_schedule_per_edge)

                











        # DEBUG 
        # print('\n----')
        # edge_to_chosen_flows = {edge: [] for edge in requested_edges.keys()}
        # for flow in chosen_flows:
            # edges = self.scheduler.get_path_edges(flow['path'])
            # for edge in edges:
                # edge_to_chosen_flows[json.dumps(sorted(edge))].append(flow['flow_id'])
        # for edge in self.scheduler.SchedulerNetwork.edges:
            # for channel in self.scheduler.RWA.channel_names:
                # bw = self.scheduler.get_channel_bandwidth(edge, channel)
                # try:
                    # print('edge: {} | channel: {} | chosen flows: {} | bandwidth remaining: {}'.format(edge, channel, edge_to_chosen_flows[json.dumps(sorted(edge))], bw))
                # except KeyError:
                    # print('edge: {} | channel: {} | chosen flows: None | bandwidth remaining: {}'.format(edge, channel, bw))




        return chosen_flows

                

    def resolve_contentions_and_set_up(self,
                                       chosen_flow, 
                                       chosen_flows, 
                                       requested_edges, 
                                       flow_id_to_cost,
                                       edge_to_sorted_costs, 
                                       edge_to_sorted_flow_ids,
                                       flow_id_to_packets_to_schedule_per_edge):
        flow = chosen_flow
        chosen_flow_ids = {f['flow_id']: None for f in chosen_flows}
        # print('\n-----')
        # print('considering flow: {}'.format(flow))
        # print('chosen flow ids: {}'.format(chosen_flow_ids))
        removed_flows = []
        loops = 0
        while True:
            if loops >= 5:
                raise Exception()
            loops += 1
            # print('flows removed:\n{}'.format(removed_flows))
            if self.scheduler.check_connection_valid(flow):
                self.scheduler.set_up_connection(flow)
                chosen_flows.append(flow)
                # print('no contention, set up correctly')
                return chosen_flows
            else:
                # print('conflict detected')
                # there's a conflict with an already chosen flow
                # find contending flow -> if has higher cost, remove it and try to establish flow again
                flow_edges = self.scheduler.get_path_edges(flow['path'])
                bandwidth_requested = (flow['packets_this_slot'] * self.scheduler.packet_size)/self.scheduler.slot_size
                # print('flow {} bandwidth requested: {}'.format(flow['flow_id'], bandwidth_requested))
                for edge in flow_edges:
                    # print(edge)
                    if not self.scheduler.check_edge_valid(flow, edge):
                        # print('found contention on edge {}'.format(edge))
                        # contention is on this edge
                        # if flow has lower cost than one of the contentions, remove highest cost (least contentious) flow and try to set up connection again
                        costs = edge_to_sorted_costs[json.dumps(sorted(edge))]
                        flow_ids = edge_to_sorted_flow_ids[json.dumps(sorted(edge))]
                        # print('flow ids requesting this edge: {}'.format(flow_ids))
                        # print('costs: {}'.format(costs))
                        for idx in reversed(range(len(flow_ids))):
                            _id = flow_ids[idx]
                            _cost = costs[idx]
                            if _id in chosen_flow_ids:
                                # this flow has previously ben chosen, check if should keep or discard
                                if _cost < flow_id_to_cost[flow['flow_id']]:
                                    # print('cost of prospective flow greater than already established flow, do not set up')
                                    # already established flow has lower cost -> do not establish flow, re-establish any flows that were taken down
                                    for f in removed_flows:
                                        self.scheduler.set_up_connection(f)
                                        chosen_flows.append(f)
                                    return chosen_flows 
                                else:
                                    # print('cost of prospective flow less than established flow, try to establish')
                                    # remove higher cost flow -> move to next while loop to try again to re-establish flow
                                    found_f = False
                                    i = 0
                                    while not found_f:
                                        f = chosen_flows[i]
                                        if f['flow_id'] == _id:
                                            # print('found high cost established flow, take down')
                                            flow['packets_this_slot'] = f['packets_this_slot'] # replace packets
                                            if flow['packets_this_slot'] > min(flow_id_to_packets_to_schedule_per_edge[flow['flow_id']]):
                                                flow['packets_this_slot'] = min(flow_id_to_packets_to_schedule_per_edge[flow['flow_id']])
                                            self.scheduler.take_down_connection(f)
                                            chosen_flows.remove(f)
                                            removed_flows.append(f)
                                            found_f = True
                                            # print('moving to next while loop iter to try set up flow again...')
                                        else:
                                            i += 1
                                    break
                            else:
                                continue 
                    else:
                        continue

    




































