from trafpy.manager.src.schedulers.schedulertoolbox import SchedulerToolbox

import json
import numpy as np
import copy
from collections import defaultdict # use for initialising arbitrary length nested dict


class FairShare:

    def __init__(self,
                 Graph,
                 RWA,
                 slot_size,
                 packet_size=300,
                 time_multiplexing=True,
                 debug_mode=False,
                 scheduler_name='fair_share'):
        self.scheduler = SchedulerToolbox(Graph, RWA, slot_size, time_multiplexing, debug_mode)
        self.scheduler_name = scheduler_name

    def get_action(self, observation, print_processing_time=False):
        chosen_flows = self.get_scheduler_action(observation)
        action = {'chosen_flows': chosen_flows}

        return action

    def get_scheduler_action(self, observation):
        # TODO: Currently assuming num_k_paths = 1. Consider how to do fair sharing where num_k_paths > 1 (N.B. Perhaps best/only way is to just randomly choose a path and a channel from k shortest paths and channels?)

        # update scheduler network & new flow states
        self.scheduler.update_network_state(observation, hide_child_dependency_flows=True)

        chosen_flows = []

        # # go through each flow and allocate it. If there's a contention, split bandwidth between all contending flows
        # for ep in self.scheduler.SchedulerNetwork.graph['endpoints']:
            # queues = self.scheduler.SchedulerNetwork.nodes[ep]
            # for queue in queues.keys():
                # for flow in queues[queue]['queued_flows']:
                    # flow = self.scheduler.init_paths_and_packets(flow)
                    # establish_flow, p, c, packets_this_slot = self.scheduler.look_for_available_lightpath(flow, chosen_flows)

                    # if establish_flow:
                        # # path and channel available, schedule flow
                        # flow['path'], flow['channel'], flow['packets_this_slot'] = p, c, packets_this_slot
                        # chosen_flows.append(flow)
                        # self.scheduler.set_up_connection(flow)

                    # else:
                        # # contention: fair share bandwidth between contending flows

                        # # find contending flows
                        # contending_flows = self.scheduler.find_all_contending_flows(flow, chosen_flows)

                        # # randomly choose a path and a channel
                        # chosen_path = np.random.choice(flow['k_shortest_paths'])
                        # chosen_channel = np.random.choice(self.scheduler.RWA.channel_names)

                        # # find which of contending flows contend this channel and collect into list
                        # contending_flows_list = contending_flows[chosen_channel]['cont_f']

                        



        # Collect dicts of i) all flows in queues and ii) which links (edges) each flow is requesting
        queued_flows = {} # maps flow_id to corresponding flow dict
        requested_edges = {} # maps links (edges) being requested to corresponding flow_ids
        for ep in self.scheduler.SchedulerNetwork.graph['endpoints']:
            queues = self.scheduler.SchedulerNetwork.nodes[ep]
            for queue in queues.keys():
                for flow in queues[queue]['queued_flows']:
                    flow = self.scheduler.init_paths_and_packets(flow)

                    # randomly choose a path and a channel
                    path_idx = np.random.choice(range(len(flow['k_shortest_paths'])))
                    flow['path'] = flow['k_shortest_paths'][path_idx]
                    flow['channel'] = np.random.choice(self.scheduler.RWA.channel_names)

                    # collect flow
                    queued_flows[flow['flow_id']] = flow

                    # collect requested edges
                    edges = self.scheduler.get_path_edges(flow['path'])
                    for e in edges:
                        if json.dumps(sorted(e)) in requested_edges.keys():
                            requested_edges[json.dumps(sorted(e))].append(flow['flow_id']) # sort to keep order consistent
                        else:
                            requested_edges[json.dumps(sorted(e))] = [flow['flow_id']] # sort to keep order consistent

        edge_to_flow_ids = {edge: [flow_id for flow_id in requested_edges[edge]] for edge in requested_edges.keys()}

        # get maximum capacity on each edge being requested
        edge_to_bandwidth = {}
        for edge in requested_edges.keys():
            try:
                bandwidth = self.scheduler.SchedulerNetwork[json.loads(edge)[0]][json.loads(edge)[1]]['max_channel_capacity']
            except KeyError:
                bandwidth = self.scheduler.SchedulerNetwork[json.loads(edge)[1]][json.loads(edge)[0]]['max_channel_capacity']
            edge_to_bandwidth[edge] = bandwidth
        
        # init fair sharing packets to schedule on each edge for each requesting flow
        edge_to_flow_id_to_packets_to_schedule = {edge:
                                                    {flow_id: 0 for flow_id in edge_to_flow_ids[edge]}
                                                  for edge in edge_to_flow_ids.keys()}

        # go through each edge and fair share available bandwidth amongst requesting flows
        for edge in requested_edges.keys():
            init_num_requests = len(requested_edges[edge])
            num_requests_left = len(requested_edges[edge])
            packets_scheduled_this_slot = 0

            # find max total packets can schedule this slot on this link
            max_info_per_slot = edge_to_bandwidth[edge] * self.scheduler.slot_size # info transferred per slot == info transferred per unit time * number of time units (i.e. slot size)
            max_packets_per_slot = int(max_info_per_slot / self.scheduler.packet_size) # round down 

            # init packets left for flows requesting this edge
            flow_packets_left = {flow_id: queued_flows[flow_id]['packets'] for flow_id in edge_to_flow_ids[edge]}

            # schedule equal packets across all 'sub slots' of slot for remaining requests/flows
            while True:
                # new sub-slot

                # find maximum possible packets can schedule for rest of time slot per remaining request
                max_packets_per_request = int((max_packets_per_slot-packets_scheduled_this_slot) / num_requests_left) # round down

                # find smallest packets left of remaining requests on this edge
                non_zero_packets = [packet for packet in flow_packets_left.values() if packet != 0]
                smallest_packets_left = min(non_zero_packets)

                # find packets to schedule per request for this sub slot
                packets_per_request = min(smallest_packets_left, max_packets_per_request)

                # fair share by schedule packets per request equally for each request for this sub-slot
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

        # packets to schedule for each flow limited by the edge in their path with lowest number of schedulable packets
        flow_id_to_packets_to_schedule_per_edge = {flow_id: [] for flow_id in queued_flows.keys()}
        for edge in edge_to_flow_id_to_packets_to_schedule.keys():
            for flow_id in edge_to_flow_id_to_packets_to_schedule[edge].keys():
                flow_id_to_packets_to_schedule_per_edge[flow_id].append(edge_to_flow_id_to_packets_to_schedule[edge][flow_id])
        for flow_id in queued_flows.keys():
            flow = queued_flows[flow_id]
            flow['packets_this_slot'] = min(flow_id_to_packets_to_schedule_per_edge[flow_id])
            chosen_flows.append(flow)

            self.scheduler.set_up_connection(flow)

        return chosen_flows





