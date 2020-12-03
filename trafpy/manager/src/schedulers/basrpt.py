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
import json
import time
import pandas as pd
from tabulate import tabulate


class BASRPT(SchedulerToolbox):

    def __init__(self, Graph, RWA, slot_size, V, packet_size=300, time_multiplexing=True, debug_mode=False, scheduler_name='basrpt'):
        super().__init__(Graph, RWA, slot_size, packet_size, time_multiplexing, debug_mode)
        self.scheduler_name = scheduler_name
        self.V = V # BASRPT V parameter
        self.N = int(len(self.Graph.graph['endpoints'])) # number of servers

        self.reset()
    
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
        flows in contending_flows will have a higher cost than this most contentious flow
        and therefore if the chosen flow has a lower cost than the most contentious flow,
        it will also have a lower cost than all competing flows and therefore should
        replace all contending flows)
        '''
        contending_flows = self.find_all_contending_flows(chosen_flow, chosen_flows, cost_metric='basrpt_cost', V=self.V, N=self.N)

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
            contending_flow, contending_flow_cost, contending_flows_list, packets_scheduled_if_drop_all_contending_flows = self.select_minimum_number_of_contending_flows_to_drop(chosen_flow, chosen_path, chosen_channel, contending_flows_list, max_packets_requested_by_chosen_flow, max_packets_available_if_all_edges_empty, 'basrpt_cost', V=self.V, N=self.N)

        num_packets_this_slot = min(max_packets_requested_by_chosen_flow, packets_scheduled_if_drop_all_contending_flows)
         
        return contending_flows_list, contending_flow, contending_flow_cost, chosen_path, chosen_channel, num_packets_this_slot 
        


    def get_scheduler_action(self, observation):
        # update scheduler network & new flow states
        self.update_network_state(observation)

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
                            contending_flows,contending_flow,contending_flow_cost,p,c,packets_this_slot = self.find_contending_flow(chosen_flow,chosen_flows)
                            chosen_flow['path'],chosen_flow['channel'], chosen_flow['packets_this_slot'] = p,c,packets_this_slot
                            chosen_cost = self.calc_basrpt_cost(chosen_flow, self.V, self.N)
                            
                            if contending_flow_cost > chosen_cost:
                                # new choice has lower cost that established flow
                                establish_flow = True
                            else:
                                # established flow has lower cost, do not choose
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
            net = self.SchedulerNetwork
            for edge in net.edges:
                for channel in net[edge[0]][edge[1]]['channels']:
                    # reset channel capacity
                    chan_cap_available = self.get_channel_bandwidth(edge, channel)
                    chan_cap_max = net[edge[0]][edge[1]]['max_channel_capacity']
                    print('Available {} capacity for {}: {} / {}'.format(channel, edge, chan_cap_available, chan_cap_max))
                            

        return chosen_flows
        

