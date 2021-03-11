from trafpy.manager.src.schedulers.schedulertoolbox import SchedulerToolbox_v2
from trafpy.manager.src.schedulers.fair_share import FairShare 
from trafpy.manager.src.schedulers.srpt import SRPT_v2

import copy





class LambdaShare:
    def __init__(self,
                 Graph,
                 RWA,
                 slot_size,
                 _lambda=0.5,
                 packet_size=300,
                 time_multiplexing=True,
                 debug_mode=False,
                 scheduler_name='\u03BBS'):
        '''
        _lambda is the share of the network bandwidth dedicated to priority
        shortest flows (i.e. this bandwidth will be given to srpt). The remaining
        bandwidth will be shared equally among all flows (i.e. this bandwidth
        will be given to fair share).

        E.g. if _lambda = 0.8 -> 80% of bandwidth used for srpt, 20% for fair share.

        '''
        # DEBUG
        # debug_mode = True
        self.debug_mode = debug_mode

        self._lambda = _lambda
        self.rwa = RWA

        # init srpt and fair share networks with lambda share bandwidth capacities
        srpt_network, fair_share_network = self._update_lambda_networks(network=Graph)

        self.srpt = SRPT_v2(Graph=srpt_network,
                            RWA=RWA,
                            slot_size=slot_size,
                            packet_size=packet_size,
                            time_multiplexing=time_multiplexing,
                            debug_mode=debug_mode)
        self.fair_share = FairShare(Graph=fair_share_network,
                                    RWA=RWA,
                                    slot_size=slot_size,
                                    packet_size=packet_size,
                                    time_multiplexing=time_multiplexing,
                                    debug_mode=debug_mode)
        self.toolbox = SchedulerToolbox_v2(Graph=Graph, 
                                           RWA=RWA, 
                                           slot_size=slot_size, 
                                           packet_size=packet_size,
                                           time_multiplexing=time_multiplexing, 
                                           debug_mode=debug_mode)

        self.scheduler_name = scheduler_name
        self.resolution_strategy='lambda_share'

    def _update_lambda_networks(self, 
                                network,
                                update_channel_capacities=True,
                                update_graph_attrs=True):
        '''
        Takes in network and initialises i) srpt and ii) fair share networks
        with updates lambda share capacities.

        if not update_channel_capacities, will not multiply channel capacities
        by lambda.

        '''
        srpt_network, fair_share_network = copy.deepcopy(network), copy.deepcopy(network)

        # update channel capacities
        if update_channel_capacities:
            for edge in network.edges:
                # max edge capacities
                # src-dst
                srpt_network[edge[0]][edge[1]]['{}_to_{}_port'.format(edge[0], edge[1])]['max_channel_capacity'] *= self._lambda
                fair_share_network[edge[0]][edge[1]]['{}_to_{}_port'.format(edge[0], edge[1])]['max_channel_capacity'] *= (1-self._lambda)
                # dst-src
                srpt_network[edge[1]][edge[0]]['{}_to_{}_port'.format(edge[1], edge[0])]['max_channel_capacity'] *= self._lambda
                fair_share_network[edge[1]][edge[0]]['{}_to_{}_port'.format(edge[1], edge[0])]['max_channel_capacity'] *= (1-self._lambda)
                for channel in self.rwa.channel_names:
                    # available edge channel capacity
                    # src-dst
                    srpt_network[edge[0]][edge[1]]['{}_to_{}_port'.format(edge[0], edge[1])]['channels'][channel] *= self._lambda
                    fair_share_network[edge[0]][edge[1]]['{}_to_{}_port'.format(edge[0], edge[1])]['channels'][channel] *= (1 - self._lambda)
                    # dst-src
                    srpt_network[edge[1]][edge[0]]['{}_to_{}_port'.format(edge[1], edge[0])]['channels'][channel] *= self._lambda
                    fair_share_network[edge[1]][edge[0]]['{}_to_{}_port'.format(edge[1], edge[0])]['channels'][channel] *= (1 - self._lambda)

        # update graph attrs
        if update_graph_attrs:
            srpt_network.graph['ep_link_capacity'] *= self._lambda
            fair_share_network.graph['ep_link_capacity'] *= (1-self._lambda)
            srpt_network.graph['ep_link_port_capacity'] *= self._lambda
            fair_share_network.graph['ep_link_port_capacity'] *= (1-self._lambda)
            srpt_network.graph['max_nw_capacity'] *= self._lambda
            fair_share_network.graph['max_nw_capacity'] *= (1-self._lambda)

        return srpt_network, fair_share_network

    def get_action(self, observation, print_processing_time=False):
        chosen_flows = self.get_scheduler_action(observation)
        return {'chosen_flows': chosen_flows}

    def get_scheduler_action(self, observation):
        if self.debug_mode:
            print('\n\n\n---------------- GET SCHEDULER ACTION -------------------')

        # update network state
        self.toolbox.update_network_state(observation, hide_child_dependency_flows=True)

        # update lambda network states
        srpt_network, fair_share_network = self._update_lambda_networks(network=observation['network'])

        # # ATTEMPT 1 & 2
        # # get srpt and fair share chosen flows
        # srpt_chosen_flows = self.srpt.get_scheduler_action(observation=srpt_network)
        # fair_share_chosen_flows = self.fair_share.get_scheduler_action(observation=fair_share_network)


        # ATTEMPT 3
        # choose fair share flows
        fair_share_chosen_flows = self.fair_share.get_scheduler_action(observation=fair_share_network, path_channel_assignment_strategy='fair_share_num_flows')

        # update srpt queued flows with chosen path and channel from fair share
        self.srpt.toolbox.update_network_state(srpt_network)
        for ep in self.fair_share.toolbox.network.graph['endpoints']:
            ep_queues = copy.deepcopy(self.fair_share.toolbox.network.nodes[ep])
            for _ep in ep_queues.keys():
                # update packets in queued flows w/ fair share choice so srpt doesn't choose flows which would otherwise be completed by fair share portion of network
                for idx in range(len(ep_queues[_ep]['queued_flows'])):
                    ep_queues[_ep]['queued_flows'][idx]['packets'] -= ep_queues[_ep]['queued_flows'][idx]['packets_this_slot']
                self.srpt.toolbox.network.nodes[ep][_ep] = ep_queues[_ep]

        # choose srpt flows, do not do any path or channel assignment (i.e. use same path and channel as allocated by fair share)
        srpt_chosen_flows = self.srpt.get_scheduler_action(observation=self.srpt.toolbox.network, reset_channel_capacities=False, path_channel_assignment_strategy=None)

        # merge fair share and srpt flows
        chosen_flows = []
        srpt_chosen_flow_id_to_flow = {flow['flow_id']: flow for flow in srpt_chosen_flows}
        fair_share_chosen_flow_id_to_flow = {flow['flow_id']: flow for flow in fair_share_chosen_flows}
        for flow_id in fair_share_chosen_flow_id_to_flow.keys():
            merged_flow = fair_share_chosen_flow_id_to_flow[flow_id]
            if flow_id in srpt_chosen_flow_id_to_flow:
                # flow was chosen by srpt, merge
                srpt_flow = srpt_chosen_flow_id_to_flow[flow_id]
                merged_flow['packets_this_slot'] += srpt_flow['packets_this_slot']
            else:
                # not chosen by srpt, no need to merge
                pass
            chosen_flows.append(merged_flow)
            self.toolbox.set_up_connection(merged_flow)




        




        # # ATTEMPT 1
        # # merge srpt and fair share chosen flows into lambda share chosen flows
        # chosen_flows = []
        # srpt_chosen_flow_id_to_flow = {flow['flow_id']: flow for flow in srpt_chosen_flows}
        # fair_share_chosen_flow_id_to_flow = {flow['flow_id']: flow for flow in fair_share_chosen_flows}
        # for flow_id in fair_share_chosen_flow_id_to_flow.keys():
            # fair_share_flow = fair_share_chosen_flow_id_to_flow[flow_id]
            # if flow_id in srpt_chosen_flow_id_to_flow:
                # # flow was chosen by srpt, merge
                # srpt_flow = srpt_chosen_flow_id_to_flow[flow_id]
                # if srpt_flow['path'] != fair_share_flow['path'] or srpt_flow['channel'] != fair_share_flow['channel']:
                    # # check which path and channel has most bw available
                    # srpt_lowest_edge_bandwidth = self.toolbox.get_lowest_edge_bandwidth(srpt_flow['path'], max_bw=False, channel=srpt_flow['channel'])
                    # fair_share_lowest_edge_bandwidth = self.toolbox.get_lowest_edge_bandwidth(fair_share_flow['path'], max_bw=False, channel=fair_share_flow['channel'])
                    # if max(srpt_lowest_edge_bandwidth, fair_share_lowest_edge_bandwidth) == 0:
                        # # no bandwidth available in either path or channel, do not set up
                        # pass
                    # else:
                        # # bandwidth available
                        # if srpt_lowest_edge_bandwidth > fair_share_lowest_edge_bandwidth:
                            # # srpt path and channel has most bw available
                            # merged_flow = copy.deepcopy(srpt_flow)
                            # max_info = srpt_lowest_edge_bandwidth * self.toolbox.slot_size 
                            # max_packets = int(max_info / self.toolbox.packet_size) # round down
                            # merged_flow['packets_this_slot'] = min(max_packets, srpt_flow['packets_this_slot'] + fair_share_flow['packets_this_slot'])
                        # else:
                            # # fair share path and channel has most bw available
                            # merged_flow = copy.deepcopy(fair_share_flow)
                            # max_info = fair_share_lowest_edge_bandwidth * self.toolbox.slot_size 
                            # max_packets = int(max_info / self.toolbox.packet_size) # round down
                            # merged_flow['packets_this_slot'] = min(max_packets, srpt_flow['packets_this_slot'] + fair_share_flow['packets_this_slot'])
                        # # set up
                        # self.toolbox.set_up_connection(merged_flow)
                        # chosen_flows.append(merged_flow)
                # else:
                    # # path and channel of srpt and fair share are the same
                    # lowest_edge_bandwidth = self.toolbox.get_lowest_edge_bandwidth(srpt_flow['path'], max_bw=False, channel=srpt_flow['channel'])
                    # if lowest_edge_bandwidth == 0:
                        # # no bandwidth available, do not set up
                        # pass
                    # else:
                        # # bandwidth available
                        # merged_flow = copy.deepcopy(srpt_flow)
                        # max_info = lowest_edge_bandwidth * self.toolbox.slot_size 
                        # max_packets = int(max_info / self.toolbox.packet_size) # round down
                        # merged_flow['packets_this_slot'] = min(max_packets, srpt_flow['packets_this_slot'] + fair_share_flow['packets_this_slot'])
                        # # set up
                        # self.toolbox.set_up_connection(merged_flow)
                        # chosen_flows.append(merged_flow)
            # else:
                # # flow not chosen by srpt, merged flow == fair share flow, set up
                # merged_flow = copy.deepcopy(fair_share_flow)
                # lowest_edge_bandwidth = self.toolbox.get_lowest_edge_bandwidth(merged_flow['path'], max_bw=False, channel=merged_flow['channel'])
                # if lowest_edge_bandwidth == 0:
                    # # no bandwidth available, do not set up
                    # pass
                # else:
                    # # bandwidth available
                    # max_info = lowest_edge_bandwidth * self.toolbox.slot_size 
                    # max_packets = int(max_info / self.toolbox.packet_size) # round down
                    # merged_flow['packets_this_slot'] = min(max_packets, merged_flow['packets_this_slot'])
                    # self.toolbox.set_up_connection(merged_flow)
                    # chosen_flows.append(merged_flow)





        # # ATTEMPT 2
        # # merge srpt and fair share chosen flows into lambda share chosen flows
        # chosen_flows = []
        # srpt_chosen_flow_id_to_flow = {flow['flow_id']: flow for flow in srpt_chosen_flows}
        # fair_share_chosen_flow_id_to_flow = {flow['flow_id']: flow for flow in fair_share_chosen_flows}
        # for flow_id in fair_share_chosen_flow_id_to_flow.keys():
            # fair_share_flow = fair_share_chosen_flow_id_to_flow[flow_id]
            # if flow_id in srpt_chosen_flow_id_to_flow:
                # # flow was chosen by srpt, merge
                # srpt_flow = srpt_chosen_flow_id_to_flow[flow_id]

                # # use fair share path and channel
                # merged_flow = copy.deepcopy(fair_share_flow)
                # lowest_edge_bandwidth = self.toolbox.get_lowest_edge_bandwidth(merged_flow['path'], max_bw=False, channel=merged_flow['channel'])
                # # srpt can only schedule up to (_lambda*bw) of fair share
                # srpt_flow['packets_this_slot'] = min(srpt_flow['packets_this_slot'],
                                                     # int(self._lambda*(fair_share_flow['packets_this_slot'] / (1-self._lambda))))
                # if lowest_edge_bandwidth != 0:
                    # # bandwidth available
                    # max_info = lowest_edge_bandwidth * self.toolbox.slot_size
                    # max_packets = int(max_info / self.toolbox.packet_size) # round down
                    # if max_packets < fair_share_flow['packets_this_slot']:
                        # # cannot schedule fair share flow packets, subtract excess
                        # srpt_packets = max_packets - fair_share_flow['packets_this_slot']
                    # elif max_packets < fair_share_flow['packets_this_slot'] + srpt_flow['packets_this_slot']:
                        # # would exceed, schedule fair share flow + max srpt packets
                        # srpt_packets = max_packets - fair_share_flow['packets_this_slot']
                    # elif max_packets == fair_share_flow['packets_this_slot']:
                        # # no extra space for srpt flow, just schedule fair_share_flow
                        # srpt_packets = 0
                    # else:
                        # # enough bw for chosen srpt packets
                        # srpt_packets = srpt_flow['packets_this_slot']
                    # # merge srpt packets with fair share packets
                    # merged_flow['packets_this_slot'] += srpt_packets
                # else:
                    # # no bandwidth available, do not schedule
                    # raise Exception('No bandwidth left')
            # else:
                # # flow not chosen by srpt, no srpt packets to merge
                # merged_flow = copy.deepcopy(fair_share_flow)
            # # set up merged flow
            # chosen_flows.append(merged_flow)
            # self.toolbox.set_up_connection(merged_flow)








            

        return chosen_flows


            


























