from trafpy.manager.src.schedulers.schedulertoolbox import SchedulerToolbox

import gym
import tensorflow as tf
import numpy as np

from ray.rllib.agents.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork

class ParametricAgent(DistributionalQTFModel):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config={},
                 name='agent',
                 true_obs_shape=(4,),
                 action_embed_size=2,
                 env = None,
                 Graph = None,
                 RWA = None,
                 slot_size = None,
                 **kw):
        super(ParametricAgent, self).__init__(obs_space, action_space, num_outputs, model_config, name, **kw)
        self.SchedulerToolbox = SchedulerToolbox(Graph, RWA, slot_size)
        print('obs_space:\n{}\naction_space:\n{}\naction_embed_space:\n{}\nmodel_config:\n{}\n'.format(obs_space, action_space, action_embed_size, model_config))
        self.action_embed_model = FullyConnectedNetwork(obs_space, action_space, action_embed_size, model_config, name+'_action_embed')
        self.register_variables(self.action_embed_model.variables())
        self.scheduler_name = name
        self.env = env

    def register_env(self, env):
        self.env = env
        self.action_space = self.env.action_space

    def forward(self, input_dict, state=None, seq_lens=None):
        if self.env is None:
            raise Exception('Must call register_env(env) method or instantiate scheduler with env != None before getting action from scheduler.')

        self.obs = input_dict['obs']
        self.SchedulerToolbox.update_network_state(self.obs, hide_child_dependency_flows=True)
        
        self.chosen_actions = []
        self.update_avail_actions(*self.chosen_actions)
        # while True:
            # self.update_avail_actions(*self.chosen_actions)
            # if len(self.avail_action_indices) > 0:
                # # still have available actions to choose from
                # pass
            # else:
                # # no available actions left
                # break

        # compute the predicted action embedding
        action_embed, _ = self.action_embed_model({'obs': self.obs['machine_readable_network']})
        
        # Expand the model output to [BATCH, 1, EMBED_SIZE]. Note that the
        # avail actions tensor is of shape [BATCH, MAX_ACTIONS, EMBED_SIZE].
        intent_vector = tf.expand_dims(action_embed, 1)

        # batch dot product -> shape of lofits is [BATCH, MAX_ACTIONS]
        action_logits = tf.reduce_sum(self.obs['machine_readable_network'] * intent_vector, axis=2)

        # mask out invalid actions
        inf_mask = tf.maximum(tf.math.log(self.action_mask), tf.float32.min)

        return action_logits + inf_mask, state

    def update_avail_actions(self, *chosen_actions):
        self.action_assignments = np.array([0.] * self.action_space.n.numpy())
        self.action_mask = np.array([0.] * self.action_space.n.numpy())

        # any actions with a path and channel that has already been chosen cannot be reselected
        chosen_flows = []
        for action in chosen_actions:
            flow = self.conv_chosen_action_index_to_chosen_flow(action, chosen_flows)
            establish_flow, path, channel = self.SchedulerToolbox.look_for_available_lightpath(flow, chosen_flows, search_k_shortest=False)
            if not establish_flow:
                raise Exception('Error: Trying to establish flow {} which is not available given chosen flows {}'.format(flow, chosen_flows))
            flow['channel'] = channel
            chosen_flows.append(flow)
        for action in self.obs['machine_readable_network'].keys():
            if self.obs['machine_readable_network'][action]['flow_present'] == 1 and self.obs['machine_readable_network'][action]['selected'] == 0 and self.obs['machine_readable_network'][action]['null_action'] == 0:
                # flow present, not yet selected and currently registered as available, check if is available given chosen actions
                flow = self.conv_chosen_action_index_to_chosen_flow(action, chosen_flows)
                establish_flow, path, channel = self.SchedulerToolbox.look_for_available_lightpath(flow, chosen_flows, search_k_shortest=False)
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
        queued_flows = self.SchedulerToolbox.SchedulerNetwork.nodes[src][dst]['queued_flows']
        found_flow = False
        for flow in queued_flows:
            if flow['src'] == src and flow['dst'] == dst and flow['time_arrived'] == time_arrived and flow['size'] == size:
                found_flow = True
                if flow['packets'] is None:
                    flow = self.SchedulerToolbox.init_paths_and_packets(flow)
                if flow['channel'] is None:
                    establish_flow, path, channel = self.SchedulerToolbox.look_for_available_lightpath(flow, chosen_flows, search_k_shortest=False)
                    if not establish_flow:
                        raise Exception('Error: Trying to establish flow {} which is not available given chosen flows {}'.format(flow, self.chosen_flows))
                    flow['channel'] = channel
                return flow
            else:
                # not this flow
                pass
        if not found_flow:
            raise Exception('Unable to find action {} in queue flows {}'.format(self.obs['machine_readable_network'][action], queued_flows))
