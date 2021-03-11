from trafpy.manager.src.schedulers.schedulertoolbox import SchedulerToolbox, SchedulerToolbox_v2

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
                 scheduler_name='FS'):
        # DEBUG
        # debug_mode = True
        self.debug_mode = debug_mode

        self.toolbox = SchedulerToolbox_v2(Graph=Graph, 
                                           RWA=RWA, 
                                           slot_size=slot_size, 
                                           packet_size=packet_size,
                                           time_multiplexing=time_multiplexing, 
                                           debug_mode=debug_mode)
        self.scheduler_name = scheduler_name
        self.resolution_strategy='fair_share'

    def get_action(self, observation, print_processing_time=False):
        chosen_flows = self.get_scheduler_action(observation)
        return {'chosen_flows': chosen_flows}

    def get_scheduler_action(self, observation, path_channel_assignment_strategy='fair_share_num_flows'):
        if self.debug_mode:
            print('\n\n\n---------------- GET SCHEDULER ACTION -------------------')

        # update network state
        self.toolbox.update_network_state(observation, hide_child_dependency_flows=True)

        # collect useful flow info dicts for making scheduling decisions
        flow_info = self.toolbox.collect_flow_info_dicts(path_channel_assignment_strategy=path_channel_assignment_strategy, cost_function=None)

        # allocate flows by sharing bandwidth equallty across all flows
        scheduling_info, cost_info = self.toolbox.allocate_available_bandwidth(flow_info, resolution_strategy=self.resolution_strategy)

        # collect chosen flows and corresponding packets to schedule for the chosen flows
        chosen_flows = []
        for flow_id in flow_info['queued_flows'].keys():
            if flow_id not in scheduling_info['flow_id_to_packets_to_schedule_per_edge'] or len(scheduling_info['flow_id_to_packets_to_schedule_per_edge'][flow_id]) == 0:
                # flow was not chosen to be scheduled on any edge
                pass
            else:
                # flow was chosen to be scheduled on an edge
                flow = flow_info['queued_flows'][flow_id]

                # packets to schedule for each flow limited by the edge in their path with lowest number of schedulable packets
                flow['packets_this_slot'] = min(scheduling_info['flow_id_to_packets_to_schedule_per_edge'][flow_id])

                # set up connection
                chosen_flows.append(flow)
                self.toolbox.set_up_connection(flow)

        # DEBUG 
        if self.debug_mode:
            print('~~~ Final Choices ~~~')
            print('chosen flows:\n{}'.format(chosen_flows))
            edge_to_chosen_flows = {edge: [] for edge in flow_info['requested_edges'].keys()}
            for flow in chosen_flows:
                edges = self.toolbox.get_path_edges(flow['path'])
                for edge in edges:
                    edge_to_chosen_flows[json.dumps(edge)].append(flow['flow_id'])
            for edge in self.toolbox.network.edges:
                for channel in self.toolbox.rwa.channel_names:
                    # src-dst
                    bw = self.toolbox.get_channel_bandwidth(edge, channel)
                    try:
                        print('edge: {} | channel: {} | chosen flows: {} | bandwidth remaining: {}'.format(edge, channel, edge_to_chosen_flows[json.dumps(edge)], bw))
                    except KeyError:
                        # no flows chosen on this edge
                        print('edge: {} | channel: {} | bandwidth remaining: {}'.format(edge, channel, bw))

                    # dst-src
                    edge = edge[::-1]
                    bw = self.toolbox.get_channel_bandwidth(edge, channel)
                    try:
                        print('edge: {} | channel: {} | chosen flows: {} | bandwidth remaining: {}'.format(edge, channel, edge_to_chosen_flows[json.dumps(edge)], bw))
                    except KeyError:
                        # no flows chosen on this edge
                        print('edge: {} | channel: {} | bandwidth remaining: {}'.format(edge, channel, bw))

        return chosen_flows



























    # def get_scheduler_action(self, observation):
        # # update scheduler network & new flow states
        # self.toolboz.update_network_state(observation, hide_child_dependency_flows=True)

        # chosen_flows = []

        # # Collect dicts of i) all flows in queues and ii) which links (edges) each flow is requesting
        # queued_flows = {} # maps flow_id to corresponding flow dict
        # requested_edges = {} # maps links (edges) being requested to corresponding flow_ids
        # for ep in self.scheduler.SchedulerNetwork.graph['endpoints']:
            # queues = self.scheduler.SchedulerNetwork.nodes[ep]
            # for queue in queues.keys():
                # for flow in queues[queue]['queued_flows']:
                    # flow = self.scheduler.init_paths_and_packets(flow)

                    # # randomly choose a path and a channel
                    # path_idx = np.random.choice(range(len(flow['k_shortest_paths'])))
                    # flow['path'] = flow['k_shortest_paths'][path_idx]
                    # flow['channel'] = np.random.choice(self.scheduler.RWA.channel_names)

                    # # collect flow
                    # queued_flows[flow['flow_id']] = flow

                    # # collect requested edges
                    # edges = self.scheduler.get_path_edges(flow['path'])
                    # for e in edges:
                        # if json.dumps(sorted(e)) in requested_edges.keys():
                            # requested_edges[json.dumps(sorted(e))].append(flow['flow_id']) # sort to keep order consistent
                        # else:
                            # requested_edges[json.dumps(sorted(e))] = [flow['flow_id']] # sort to keep order consistent

        # edge_to_flow_ids = {edge: [flow_id for flow_id in requested_edges[edge]] for edge in requested_edges.keys()}

        # # get maximum capacity on each edge being requested
        # edge_to_bandwidth = {}
        # for edge in requested_edges.keys():
            # try:
                # bandwidth = self.scheduler.SchedulerNetwork[json.loads(edge)[0]][json.loads(edge)[1]]['max_channel_capacity']
            # except KeyError:
                # bandwidth = self.scheduler.SchedulerNetwork[json.loads(edge)[1]][json.loads(edge)[0]]['max_channel_capacity']
            # edge_to_bandwidth[edge] = bandwidth
        
        # # init fair sharing packets to schedule on each edge for each requesting flow
        # edge_to_flow_id_to_packets_to_schedule = {edge:
                                                    # {flow_id: 0 for flow_id in edge_to_flow_ids[edge]}
                                                  # for edge in edge_to_flow_ids.keys()}

        # # go through each edge and fair share available bandwidth amongst requesting flows
        # for edge in requested_edges.keys():
            # init_num_requests = len(requested_edges[edge])
            # num_requests_left = len(requested_edges[edge])
            # packets_scheduled_this_slot = 0

            # # find max total packets can schedule this slot on this link
            # max_info_per_slot = edge_to_bandwidth[edge] * self.scheduler.slot_size # info transferred per slot == info transferred per unit time * number of time units (i.e. slot size)
            # max_packets_per_slot = int(max_info_per_slot / self.scheduler.packet_size) # round down 

            # # init packets left for flows requesting this edge
            # flow_packets_left = {flow_id: queued_flows[flow_id]['packets'] for flow_id in edge_to_flow_ids[edge]}

            # # schedule equal packets across all 'sub slots' of slot for remaining requests/flows
            # while True:
                # # new sub-slot

                # # find maximum possible packets can schedule for rest of time slot per remaining request
                # max_packets_per_request = int((max_packets_per_slot-packets_scheduled_this_slot) / num_requests_left) # round down

                # # find smallest packets left of remaining requests on this edge
                # non_zero_packets = [packet for packet in flow_packets_left.values() if packet != 0]
                # smallest_packets_left = min(non_zero_packets)

                # # find packets to schedule per request for this sub slot
                # packets_per_request = min(smallest_packets_left, max_packets_per_request)

                # # fair share by schedule packets per request equally for each request for this sub-slot
                # for flow_id in flow_packets_left.keys():
                    # if flow_packets_left[flow_id] != 0:
                        # edge_to_flow_id_to_packets_to_schedule[edge][flow_id] += packets_per_request
                        # flow_packets_left[flow_id] -= packets_per_request
                        # packets_scheduled_this_slot += packets_per_request
                        # if flow_packets_left[flow_id] == 0:
                            # # all packets of this flow have now been scheduled
                            # num_requests_left -= 1
                
                # if packets_scheduled_this_slot >= max_packets_per_slot-init_num_requests or num_requests_left == 0:
                    # # finished fair sharing this slot for this edge
                    # break
                # else:
                    # # move to next sub-slot for this edge
                    # pass

        # # packets to schedule for each flow limited by the edge in their path with lowest number of schedulable packets
        # flow_id_to_packets_to_schedule_per_edge = {flow_id: [] for flow_id in queued_flows.keys()}
        # for edge in edge_to_flow_id_to_packets_to_schedule.keys():
            # for flow_id in edge_to_flow_id_to_packets_to_schedule[edge].keys():
                # flow_id_to_packets_to_schedule_per_edge[flow_id].append(edge_to_flow_id_to_packets_to_schedule[edge][flow_id])
        # for flow_id in queued_flows.keys():
            # flow = queued_flows[flow_id]
            # flow['packets_this_slot'] = min(flow_id_to_packets_to_schedule_per_edge[flow_id])
            # chosen_flows.append(flow)

            # self.scheduler.set_up_connection(flow)

        # return chosen_flows















