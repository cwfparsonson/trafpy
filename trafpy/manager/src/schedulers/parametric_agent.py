import gym
import tensorflow as tf
from ray.rllib.agents.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork

class ParametricAgent(DistributionalQTFModel):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 true_obs_shape=(4,),
                 action_embed_size=2,
                 **kw):
        super(ParametricAgent, self).__init__(obs_space, action_space, num_outputs, model_config, name, **kw)
        self.action_embed_model = FullyConnectedNetwork(gym.spaces.Box(-1, 1e12, shape=true_obs_shape), action_space, action_embed_size, model_config, name+'_action_embed')
        self.register_variables(self.action_embed_model.variables())

    def forward(self, input_dict, state, seq_lens):
        # extract available actions tensor from observation
        avail_actions = input_dict['obs']['avail_actions']
        action_mask = input_dict['obs']['action_mask']

        # compute the predicted action embedding
        action_embed, _ = self.action_embed_model({'obs': input_dict['obs']['machine_readable_network']})
        
        # Expand the model output to [BATCH, 1, EMBED_SIZE]. Note that the
        # avail actions tensor is of shape [BATCH, MAX_ACTIONS, EMBED_SIZE].
        intent_vector = tf.expand_dims(action_embed, 1)

        # batch dot product -> shape of lofits is [BATCH, MAX_ACTIONS]
        action_logits = tf.reduce_sum(avail_actions * intent_vector, axis=2)

        # mask out invalid actions
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)

        return action_logits + inf_mask, state

