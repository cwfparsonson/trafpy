from trafpy.manager.src.schedulers.schedulertoolbox import SchedulerToolbox, SchedulerToolbox_v2

import random
import json


class RandomAgent:

    def __init__(self,
                 Graph,
                 RWA,
                 slot_size,
                 packet_size=300,
                 time_multiplexing=True,
                 debug_mode=False,
                 scheduler_name='Rand'):
        self.debug_mode = debug_mode
        self.toolbox = SchedulerToolbox_v2(Graph=Graph, 
                                           RWA=RWA, 
                                           slot_size=slot_size, 
                                           packet_size=packet_size,
                                           time_multiplexing=time_multiplexing, 
                                           debug_mode=debug_mode)
        self.scheduler_name = scheduler_name
        self.resolution_strategy = 'random'


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


































































# OLD (DON'T DELETE: This uses machine_readable_network observation -> might be useful when implementing RL stuff
# import numpy as np
# import json
# import tensorflow as tf

# class RandomAgent(SchedulerToolbox):

    # def __init__(self, Graph, RWA, slot_size, packet_size=300, env=None, scheduler_name='random'):
        # super().__init__(Graph, RWA, slot_size, packet_size)
        # self.env = env
        # self.scheduler_name = scheduler_name

    # def update_avail_actions(self, *chosen_actions):
        # self.action_assignments = np.array([0.] * self.action_space.n.numpy())
        # self.action_mask = np.array([0.] * self.action_space.n.numpy())

        # # any actions with a path and channel that has already been chosen cannot be reselected
        # chosen_flows = []
        # for action in chosen_actions:
            # flow = self.conv_chosen_action_index_to_chosen_flow(action, chosen_flows)
            # establish_flow, path, channel = self.look_for_available_lightpath(flow, chosen_flows, search_k_shortest=False)
            # if not establish_flow:
                # raise Exception('Error: Trying to establish flow {} which is not available given chosen flows {}'.format(flow, chosen_flows))
            # flow['channel'] = channel
            # chosen_flows.append(flow)
        # for action in self.obs['machine_readable_network'].keys():
            # if self.obs['machine_readable_network'][action]['flow_present'] == 1 and self.obs['machine_readable_network'][action]['selected'] == 0 and self.obs['machine_readable_network'][action]['null_action'] == 0:
                # # flow present, not yet selected and currently registered as available, check if is available given chosen actions
                # flow = self.conv_chosen_action_index_to_chosen_flow(action, chosen_flows)
                # establish_flow, path, channel = self.look_for_available_lightpath(flow, chosen_flows, search_k_shortest=False)
                # if not establish_flow:
                    # # no way to establish flow, register as null action
                    # self.obs['machine_readable_network'][action]['null_action'] = 1

        # # get indices of available actions and create action mask
        # self.avail_action_indices = self.get_indices_of_available_actions(self.obs['machine_readable_network'])
        # for i in self.avail_action_indices:
            # self.action_mask[i] = 1

    # def get_indices_of_available_actions(self, action_dict):
        # indices = []
        # for index in action_dict.keys():
            # if action_dict[index]['flow_present'] == 1 and action_dict[index]['selected'] == 0 and action_dict[index]['null_action'] == 0:
                # # flow present and not yet selected
                # indices.append(index)
            # else:
                # # no flow present at this placeholder action
                # pass

        # return indices

    # def get_scheduler_action(self, obs, choose_multiple_actions=True):
        # self.obs = obs
        # self.update_network_state(obs, hide_child_dependency_flows=True)

        # self.chosen_actions = []
        # if choose_multiple_actions:
            # while True:
                # # choose flows
                # self.update_avail_actions(*self.chosen_actions)
                # if len(self.avail_action_indices) > 0:
                    # # still have available actions to choose from
                    # action = np.random.choice(self.avail_action_indices)

                    # # add sampled action to chosen actions
                    # self.chosen_actions.append(action)

                    # # update action as having been chosen
                    # self.obs['machine_readable_network'][action]['selected'] = 1
                    # self.obs['machine_readable_network'][action]['null_action'] = 1

                # else:
                    # # no available actions left
                    # break

        # else:
            # self.update_avail_actions(*self.chosen_actions)
            # action = np.random.choice(self.avail_action_indices)
            # self.chosen_actions.append(action)
            # self.obs['machine_readable_network'][action]['selected'] = 1
            # self.obs['machine_readable_network'][action]['null_action'] = 1

        # self.chosen_flows = []
        # for action in self.chosen_actions:
            # self.chosen_flows.append(self.conv_chosen_action_index_to_chosen_flow(action))

        # return self.chosen_flows

    # def conv_chosen_action_index_to_chosen_flow(self, action, chosen_flows=None):
        # if chosen_flows is None:
            # chosen_flows = self.chosen_flows
        # else:
            # pass
        # # find src and dst of chosen action flow
        # src_rep = self.obs['machine_readable_network'][action]['src']
        # dst_rep = self.obs['machine_readable_network'][action]['dst']
        # src = self.env.repgen.index_to_endpoint[src_rep.numpy()]
        # dst = self.env.repgen.index_to_endpoint[dst_rep.numpy()]
        # # find other unique chars of flow
        # time_arrived = self.obs['machine_readable_network'][action]['time_arrived']
        # size = self.obs['machine_readable_network'][action]['size']
        # # find queued flows at this src-dst queue
        # queued_flows = self.SchedulerNetwork.nodes[src][dst]['queued_flows']
        # found_flow = False
        # for flow in queued_flows:
            # if flow['src'] == src and flow['dst'] == dst and flow['time_arrived'] == time_arrived and flow['size'] == size:
                # found_flow = True
                # if flow['packets'] is None:
                    # flow = self.init_paths_and_packets(flow)
                # if flow['channel'] is None:
                    # establish_flow, path, channel = self.look_for_available_lightpath(flow, chosen_flows, search_k_shortest=False)
                    # if not establish_flow:
                        # raise Exception('Error: Trying to establish flow {} which is not available given chosen flows {}'.format(flow, self.chosen_flows))
                    # flow['channel'] = channel
                # return flow
            # else:
                # # not this flow
                # pass
        # if not found_flow:
            # raise Exception('Unable to find action {} in queue flows {}'.format(self.obs['machine_readable_network'][action], queued_flows))

    # def register_env(self, env):
        # self.env = env
        # self.action_space = self.env.action_space

    # def get_action(self, obs):
        # if self.env is None:
            # raise Exception('Must call register_env(env) method or instantiate scheduler with env != None before getting action from scheduler.')
        # chosen_flows = self.get_scheduler_action(obs)
        # action = {'chosen_flows': chosen_flows}

        # return action



