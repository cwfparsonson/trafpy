from trafpy.manager.src.schedulers.schedulertoolbox import SchedulerToolbox_v2

import json




class FirstFit:

    def __init__(self, 
                 Graph, 
                 RWA, 
                 slot_size, 
                 packet_size=300, 
                 time_multiplexing=True, 
                 debug_mode=False, 
                 scheduler_name='FF'):
        self.debug_mode = debug_mode
        self.toolbox = SchedulerToolbox_v2(Graph=Graph, 
                                           RWA=RWA, 
                                           slot_size=slot_size, 
                                           packet_size=packet_size,
                                           time_multiplexing=time_multiplexing, 
                                           debug_mode=debug_mode)
        self.scheduler_name = scheduler_name
        self.resolution_strategy = 'first_fit'


    def get_action(self, observation, print_processing_time=False):
        chosen_flows = self.get_scheduler_action(observation)
        return {'chosen_flows': chosen_flows}

    def get_scheduler_action(self, observation):
        # update network state
        self.toolbox.update_network_state(observation, hide_child_dependency_flows=True)

        # collect useful flow info dicts for making scheduling decisions
        flow_info = self.toolbox.collect_flow_info_dicts(path_channel_assignment_strategy='fair_share_num_flows', cost_function=None)

        # allocate flows by order of cost (lowest cost flows prioritised first)
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

                # check that flow was also selected to be scheduled on a bandwidth-limited end point link (if it wasn't, cannot schedule this flow)
                info_to_transfer_this_slot = flow['packets_this_slot'] * flow['packet_size']
                lowest_edge_bandwidth = self.toolbox.get_lowest_edge_bandwidth(flow['path'])
                bandwidth_requested = info_to_transfer_this_slot / self.toolbox.slot_size # info units of this flow transferred this time slot == capacity used on each channel in flow's path this time slot
                if bandwidth_requested > lowest_edge_bandwidth:
                    # flow must only have been selected for bandwidth-limiting links and not been given any bandwidth-limited capacity, do not schedule flow
                    pass
                else:
                    # flow must have been allocated bandwidth on at least one end point link, check for contentions and try to establish flow 
                    chosen_flows = self.toolbox.resolve_contentions_and_set_up_flow(flow, chosen_flows, flow_info, scheduling_info, cost_info, resolution_strategy=self.resolution_strategy)

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













































