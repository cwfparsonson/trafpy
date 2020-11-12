# TODO: Get rand agent choosing 1 flow per time slot
# TODO: Get rand agent choosing multiple flows per time slot
# TODO: Get rand agent choosing a chosen number of flows per time slot (until select null action or until invalid)

from trafpy.manager.src.schedulers.schedulertoolbox import SchedulerToolbox

import numpy as np
import json
import tensorflow as tf



class RandomAgent(SchedulerToolbox):

    def __init__(self, Graph, RWA, slot_size, env, scheduler_name='random'):
        super().__init__(Graph, RWA, slot_size)
        self.env = env
        self.action_space = self.env.action_space
        self.scheduler_name = scheduler_name

    def update_avail_actions(self):
        self.action_assignments = np.array([0.] * self.action_space.n.numpy())
        self.action_mask = np.array([0.] * self.action_space.n.numpy())

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

    

    def get_scheduler_action(self, obs, choose_multiple_actions=False):
        self.obs = obs
        self.update_network_state(obs, hide_child_dependency_flows=True)

        chosen_actions = []

        if choose_multiple_actions:
            while True:
                # choose flows
                self.update_avail_actions()
                if len(self.avail_action_indices) > 0:
                    # still have available actions to choose from
                    action = self.env.action_space.sample()
                    while action not in self.avail_action_indices:
                        # sample an action until get one that is available
                        action = self.env.action_space.sample()

                    # add sampled action to chosen actions
                    chosen_actions.append(action)

                    # update action as having been chosen
                    self.obs['machine_readable_network'][action]['selected'] = 1
                    self.obs['machine_readable_network'][action]['null_action'] = 1

                else:
                    # no available actions left
                    break

        else:
            self.update_avail_actions()
            action = self.env.action_space.sample()
            while action not in self.avail_action_indices:
                action = self.env.action_space.sample()
            chosen_actions.append(action)

        return self.conv_chosen_action_indices_to_chosen_flows(chosen_actions)




    def conv_chosen_action_indices_to_chosen_flows(self, chosen_actions):
            # conv chosen actions to chosen flows
            chosen_flows = []
            for action in chosen_actions:
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
                            _, packets = self.gen_flow_packets(flow['size'])
                            flow['packets'] = packets
                        chosen_flows.append(flow)
                    else:
                        # not this flow
                        pass
                if not found_flow:
                    raise Exception('Unable to find action {} in queue flows {}'.format(self.obs['machine_readable_network'][action], queued_flows))

            return chosen_flows





        


    

    def get_action(self, obs):
        chosen_flows = self.get_scheduler_action(obs)
        action = {'chosen_flows': chosen_flows}

        return action



