# TODO: Get rand agent choosing multiple flows per time slot
# TODO: Get rand agent choosing a chosen number of flows per time slot (until select null action or until invalid)

from trafpy.manager.src.schedulers.schedulertoolbox import SchedulerToolbox

import numpy as np
import json
import tensorflow as tf

class RandomAgent(SchedulerToolbox):

    def __init__(self, Graph, RWA, slot_size, env=None, scheduler_name='random'):
        super().__init__(Graph, RWA, slot_size)
        self.env = env
        self.scheduler_name = scheduler_name

    def update_avail_actions(self, *chosen_actions):
        self.action_assignments = np.array([0.] * self.action_space.n.numpy())
        self.action_mask = np.array([0.] * self.action_space.n.numpy())

        # any actions with a path and channel that has already been chosen cannot be reselected
        chosen_flows = []
        for action in chosen_actions:
            flow = self.conv_chosen_action_index_to_chosen_flow(action, chosen_flows)
            establish_flow, path, channel = self.look_for_available_lightpath(flow, chosen_flows, search_k_shortest=False)
            if not establish_flow:
                raise Exception('Error: Trying to establish flow {} which is not available given chosen flows {}'.format(flow, chosen_flows))
            flow['channel'] = channel
            chosen_flows.append(flow)
        for action in self.obs['machine_readable_network'].keys():
            if self.obs['machine_readable_network'][action]['flow_present'] == 1 and self.obs['machine_readable_network'][action]['selected'] == 0 and self.obs['machine_readable_network'][action]['null_action'] == 0:
                # flow present, not yet selected and currently registered as available, check if is available given chosen actions
                flow = self.conv_chosen_action_index_to_chosen_flow(action, chosen_flows)
                establish_flow, path, channel = self.look_for_available_lightpath(flow, chosen_flows, search_k_shortest=False)
                if not establish_flow:
                    # no way to establish flow, register as null action
                    self.obs['machine_readable_network'][action]['null_action'] = 1

        # get indices of available actions and create action mask
        self.avail_action_indices = self.get_indices_of_available_actions(self.obs['machine_readable_network'])
        for i in self.avail_action_indices:
            self.action_mask[i] = 1

    def get_indices_of_available_actions(self, action_dict):
        indices = []
        for index in action_dict.keys():
            if action_dict[index]['flow_present'] == 1 and action_dict[index]['selected'] == 0 and action_dict[index]['null_action'] == 0:
                # flow present and not yet selected
                indices.append(index)
            else:
                # no flow present at this placeholder action
                pass

        return indices

    def get_scheduler_action(self, obs, choose_multiple_actions=True):
        self.obs = obs
        self.update_network_state(obs, hide_child_dependency_flows=True)

        self.chosen_actions = []
        if choose_multiple_actions:
            while True:
                # choose flows
                self.update_avail_actions(*self.chosen_actions)
                if len(self.avail_action_indices) > 0:
                    # still have available actions to choose from
                    action = np.random.choice(self.avail_action_indices)

                    # add sampled action to chosen actions
                    self.chosen_actions.append(action)

                    # update action as having been chosen
                    self.obs['machine_readable_network'][action]['selected'] = 1
                    self.obs['machine_readable_network'][action]['null_action'] = 1

                else:
                    # no available actions left
                    break

        else:
            self.update_avail_actions(*self.chosen_actions)
            action = np.random.choice(self.avail_action_indices)
            self.chosen_actions.append(action)
            self.obs['machine_readable_network'][action]['selected'] = 1
            self.obs['machine_readable_network'][action]['null_action'] = 1

        self.chosen_flows = []
        for action in self.chosen_actions:
            self.chosen_flows.append(self.conv_chosen_action_index_to_chosen_flow(action))

        return self.chosen_flows

    def conv_chosen_action_index_to_chosen_flow(self, action, chosen_flows=None):
        if chosen_flows is None:
            chosen_flows = self.chosen_flows
        else:
            pass
        # find src and dst of chosen action flow
        src_rep = self.obs['machine_readable_network'][action]['src']
        dst_rep = self.obs['machine_readable_network'][action]['dst']
        src = self.env.repgen.index_to_endpoint[src_rep.numpy()]
        dst = self.env.repgen.index_to_endpoint[dst_rep.numpy()]
        # find other unique chars of flow
        time_arrived = self.obs['machine_readable_network'][action]['time_arrived']
        size = self.obs['machine_readable_network'][action]['size']
        # find queued flows at this src-dst queue
        queued_flows = self.SchedulerNetwork.nodes[src][dst]['queued_flows']
        found_flow = False
        for flow in queued_flows:
            if flow['src'] == src and flow['dst'] == dst and flow['time_arrived'] == time_arrived and flow['size'] == size:
                found_flow = True
                if flow['packets'] is None:
                    flow = self.init_paths_and_packets(flow)
                if flow['channel'] is None:
                    establish_flow, path, channel = self.look_for_available_lightpath(flow, chosen_flows, search_k_shortest=False)
                    if not establish_flow:
                        raise Exception('Error: Trying to establish flow {} which is not available given chosen flows {}'.format(flow, self.chosen_flows))
                    flow['channel'] = channel
                return flow
            else:
                # not this flow
                pass
        if not found_flow:
            raise Exception('Unable to find action {} in queue flows {}'.format(self.obs['machine_readable_network'][action], queued_flows))

    def register_env(self, env):
        self.env = env
        self.action_space = self.env.action_space

    def get_action(self, obs):
        if self.env is None:
            raise Exception('Must call register_env(env) method or instantiate scheduler with env != None before getting action from scheduler.')
        chosen_flows = self.get_scheduler_action(obs)
        action = {'chosen_flows': chosen_flows}

        return action



