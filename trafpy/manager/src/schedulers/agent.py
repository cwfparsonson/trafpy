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


class Agent(SchedulerToolbox):

    def __init__(self, Graph, RWA, slot_size, max_F, epsilon, alpha, gamma, agent_type='sarsa_learning'):
        super().__init__(Graph, RWA, slot_size)
        self.agent_type = agent_type
        self.N = int(len(self.Graph.graph['endpoints'])) # number of servers
        
        self.max_F = max_F # max number of flows per queue
        self.max_record_num_packets = 100 # max to record. above this val, all same
        self.max_record_time_in_queue = 10 # max time in queue to record
        
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        graph_edges = [e for e in self.Graph.edges]
        self.num_edges = len(graph_edges)
        self.num_encoded_path_combos = self.RWA.num_k * (2 ** (self.num_edges-1)) 

        self.num_poss_times_in_queue = (self.max_record_time_in_queue/self.slot_size)+1
        self.num_poss_schedule = 2 # can be 1 or 0
        self.num_poss_packets = self.max_record_num_packets + 1
        self.num_poss_flows = self.max_F + 1
        #self.phi = self.num_poss_packets * self.num_poss_times_in_queue * self.num_poss_flows * self.num_encoded_path_combos * self.num_poss_schedule # state space for each flow 
        self.phi = int(self.num_poss_packets + self.num_poss_times_in_queue + self.num_poss_flows + (self.RWA.num_k * self.num_edges) + 1) # num params for each flow state, +1 is for schedule param 

        self.num_queues = self.N * (self.N - 1)
        eps = self.Graph.graph['endpoints']
        a_idx = iter([i+1 for i in range(self.num_queues * self.max_F)])
        self.int_to_action = {next(a_idx): {'server': ep,
                                            'queue': epq,
                                            'flow_idx': flow_idx}
                                for ep in eps
                                    for epq in eps if ep != epq
                                        for flow_idx in range(self.max_F)}
        self.int_to_action[0] = 'null' # add null action

        #self.action_to_int = {}
        #for key, val in self.int_to_action.items():
        #    self.action_to_int[val] = key
        
        self.reset()
        
        self.action_space = len(self.int_to_action)
        self.state_space = self.num_queues * self.max_F * self.phi
        #self.state_space = (self.action_space ** 2) * self.phi
        
        print('Action space: {}'.format(self.action_space))
        print('State space: {}'.format(self.state_space))
        
        # initialise q table
        self.Q_table = defaultdict(lambda: np.zeros(self.action_space))
        
        # create policy to follow
        self.policy = self.make_epsilon_greedy_policy(self.Q_table, self.epsilon, self.action_space)

    def make_epsilon_greedy_policy(self, Q_table, epsilon, action_space):
        def policy_func(state):
            # create uniform probability distribution over all possible actions
            A = np.ones(action_space, dtype=float) * epsilon / action_space
            
            # find which action has the highest value
            best_action = np.argmax(Q_table[state])
            print('Best action:\n{}'.format(best_action))
            
            # bias probability of choosing best action
            A[best_action] += (1.0 - epsilon)
            print('Action prob distribution:\n{}'.format(A))

            return A

        return policy_func

    def check_if_end_of_time_slot_decisions(self, action, chosen_flows):
        '''
        If one of the following is true, agent should stop making decisions
        this time slot and this method will return True:

        1) The action chosen is the 'null' action, which is the agent's way
        of explicitly stating that it doesn't want to schedule anymore flows
        for this time slot

        2) The action chosen is a flow that has already been chosen this time
        slot

        3) The action chosen is invalid since, due to previously selected 
        actions, there are no paths or channels (lightpaths) available

        Args:
        - action: The action to be checked chosen by the agent
        - chosen_flows: A list of flows already chosen by the agent
        '''
        if action == 'null':
            print('Agent doesn\'t want to make any more scheduling decisions')
            return True

        else:
            # find flow in network
            flow_dict = self.get_flow_from_network(action['server'],
                                                   action['queue'],
                                                   action['flow_idx'])
            if flow_dict == 'N/A':
                print('Flow not in network, therefore action invalid')
                return True
            else:
                # flow is in network, check if already chosen this time slot
                if flow_dict in chosen_flows:
                    print('Chosen flow already chosen this time slot')
                    return True
                else:
                    # flow not yet chosen this time slot, check if lightpath available
                    est, _, _ = self.look_for_available_lightpath(flow_dict, chosen_flows)
                    if not est:
                        print('Lightpath not available, therefore action invalid')
                        return True
        
        # if get to this point, action not null and is valid, return False
        return False


    def get_action(self, observation):
        chosen_flows = []
        self.chosen_actions = [] # save chosen actions
        self.chosen_action_states = [] # save states agent observed when choosing flows
        self.Q_state_action = 0 # init estimated q state action value

        self.update_network_state(observation)
        state, agent_queues = self.get_agent_state_representation(observation) 
        
        while True:
            # keep getting decisions until end of time slot decisions
            a_probs = self.policy(state) 
            a = np.random.choice(np.arange(len(a_probs)),p=a_probs) # choose from dist
            print('Chosen action: {}'.format(a))
            self.chosen_actions.append(a)
            self.chosen_action_states.append(state)
            q_value = self.Q_table[state][a]
            print('Estimated q_value of action: {}'.format(q_value))
            self.Q_state_action += q_value


            a_meaning = self.int_to_action[a]
            print('Action meaning:\n{}'.format(a_meaning))
            
            # check if chosen action is null or invalid
            if self.check_if_end_of_time_slot_decisions(a_meaning, chosen_flows):
                # agent has either chosen 'null' or an invalid action
                print('No more scheduling decisions for this time step')
                print('Chosen flows:\n{}'.format(chosen_flows))
                
                # save current observation and actions chosen
                action = {'chosen_flows': chosen_flows}
                self.curr_observation = copy.deepcopy(observation)
                self.curr_action = copy.deepcopy(action)
                
                # update Q_state_action to be estimated mean return per action
                num_chosen_actions = len(chosen_flows)
                try:
                    self.Q_state_action /= num_chosen_actions
                except ZeroDivisionError:
                    # estimated 0 mean return per action
                    self.Q_state_action

                return action 
            else:
                # agent action is valid. Allocate lightpath with FF and append to chosen flows
                chosen_flow = self.get_flow_from_network(a_meaning['server'], 
                                                         a_meaning['queue'],
                                                         a_meaning['flow_idx'])
                _, p, c = self.look_for_available_lightpath(chosen_flow,
                                                            chosen_flows)
                chosen_flow['path'], chosen_flow['channel'] = p, c
                chosen_flows.append(chosen_flow)

                # update agent state as having scheduled this chosen flow
                state, agent_queues = self.update_agent_state(agent_queues, a_meaning)
    
    def update_agent_state(self, agent_queues, action):
        '''
        Updates flow=action in agent_queues to having scheduled = 1, returns
        updated state
        '''
        server = action['server']
        queue = action['queue']
        flow_idx = action['flow_idx']

        # update schedule status of flow in agent queues
        #print('Agent queue:\n{}'.format(agent_queues[server][queue]['queued_flows'][flow_idx]))
        agent_queues[server][queue]['queued_flows'][flow_idx]['scheduled'] = [int(1)]
        #print('Updated agent queues:\n{}'.format(agent_queues))

        # update state
        state = self.gen_state_from_agent_queues(agent_queues)
        hashable_state=tuple(list(itertools.chain.from_iterable(state))) # merge list of lists

        return hashable_state, agent_queues



    def process_reward(self, reward, next_observation):
        '''
        Take reward from environment that resulted in action from prev time step
        and use to learn 
        '''
        # save what your estimated value per chosen flow/action was
        Q_state_action = copy.deepcopy(self.Q_state_action)

        # get mean value per action for next observation using best actions
        #saved_epsilon = copy.deepcopy(self.epsilon) # save curr epsilon val
        #self.epsilon = 0 # choose best actions
        _ = self.get_action(next_observation)
        Q_nextstate_nextaction = copy.deepcopy(self.Q_state_action)
        #self.epsilon = saved_epsilon # return to saved epsilon val

        # calc td delta
        td_target = reward + (self.gamma * (Q_nextstate_nextaction))
        td_delta = td_target - Q_state_action
        
        # go through each action that was chosen and update q value
        iterables = zip(self.chosen_actions, self.chosen_action_states)
        for action, state in iterables:
            self.Q_table[state][action] += self.alpha * td_delta


        




    def get_agent_state_representation(self, observation):
        eps = self.SchedulerNetwork.graph['endpoints']
        agent_queues = {server: {queue: {'queued_flows': []}
                            for queue in eps if queue != server}
                                for server in eps}
        
        
        for ep in eps:
            # check queues at each end point...
            ep_queues = self.SchedulerNetwork.nodes[ep]
            agent_ep_queue = {ep: {'queued_flows': []}}
            iterables = zip(ep_queues.values(), ep_queues.keys())
            for ep_queue, ep_queue_key in iterables:
                # for each queue at the selected end point...
                
                # calc num flows in queue
                num_flows_in_queue = len(ep_queue['queued_flows'])
                encoded_num_flows = self.binary_encode_num_flows_in_queue(num_flows_in_queue,
                                                                          self.max_F)

                for flow_idx in range(num_flows_in_queue):
                    # for each flow in the selected queue...
                    flow_dict = self.init_paths_and_packets(ep_queue['queued_flows'][flow_idx])

                    # calc time flow has been in queue so far
                    num_decimals = str(self.slot_size)[::-1].find('.')
                    time_arrived = flow_dict['time_arrived']
                    curr_time = observation['slot_dict']['lb_time']
                    time_in_queue = abs(round(curr_time - time_arrived, num_decimals)) 
                    encoded_time = self.binary_encode_time_in_queue(time_in_queue,
                                                                    self.max_record_time_in_queue)

                    # calc num flow packets remaining
                    num_packets_left = len(flow_dict['packets'])
                    encoded_num_packets = self.binary_encode_num_packets(num_packets_left,
                                                                         self.max_record_num_packets)

                    # get binary encoded shortest paths
                    shortest_paths = flow_dict['k_shortest_paths']
                    encoded_paths = self.binary_encode_paths(shortest_paths)

                    # init scheduled status for this time slot
                    scheduled = [0]

                    # define the agent's flow dict
                    agent_flow_dict = {'num_packets_left': encoded_num_packets,
                                       'time_in_queue': encoded_time,
                                       'num_flows_in_queue': encoded_num_flows,
                                       'k_shortest_paths': encoded_paths,
                                       'scheduled': scheduled}
                    
                    # add flow dict to agent's queue
                    agent_queues[ep][ep_queue_key]['queued_flows'].append(agent_flow_dict)
                    # agent_ep_queue[ep]['queued_flows'].append(agent_flow_dict)

            # update agent's queue state for this end point...
            #agent_queues[next(key_iter)] = agent_ep_queues
            

        state = self.gen_state_from_agent_queues(agent_queues)
        
        hashable_state=tuple(list(itertools.chain.from_iterable(state))) # merge list of lists

        return hashable_state, agent_queues

    def merge_agent_flow_dict(self, agent_flow_dict):
        '''
        Merges flow dict of agent into single array
        '''
        flow_state_array = []
        flow_state_array.append(agent_flow_dict['scheduled'])
        flow_state_array.append(agent_flow_dict['num_packets_left'])
        flow_state_array.append(agent_flow_dict['time_in_queue'])
        flow_state_array.append(agent_flow_dict['num_flows_in_queue'])
        paths = list(itertools.chain.from_iterable(agent_flow_dict['k_shortest_paths']))
        flow_state_array.append(paths)
        flow_state_array=list(itertools.chain.from_iterable(flow_state_array)) # merge list of lists 
       
        return flow_state_array


    def gen_state_from_agent_queues(self, agent_queues):
        '''
        Uses agent queues to generate state
        '''
        # init state for single flow
        single_flow_state = list(np.zeros((self.phi,), dtype=int))
        #print('len single flow state: {}'.format(len(single_flow_state)))

        # calc total number of flows in network state space
        num_flows = self.num_queues * self.max_F

        # merge agent queue states that have been added into single array
        state_queue_dict = copy.deepcopy(agent_queues)
        for ep in state_queue_dict.keys():
            queues = state_queue_dict[ep]
            for queue in queues.keys():
                flows = state_queue_dict[ep][queue]['queued_flows']
                for idx in range(len(flows)):
                    flows[idx] = self.merge_agent_flow_dict(flows[idx])
                    #print('len merged flow: {}'.format(len(flows[idx])))
        #print('Original agent queues:\n{}'.format(agent_queues))
        #print('Merged agent queues:\n{}'.format(state_queue_dict))

        # each queue of each server can have a flow, where a flow has self.phi parameters
        # add initialised single flow state for each flow until have padded out state dict
        for ep in state_queue_dict.keys():
            queues = state_queue_dict[ep]
            for queue in queues.keys():
                flows = state_queue_dict[ep][queue]['queued_flows']
                while len(flows) < self.max_F:
                    flows.append(single_flow_state)
        #print('Padded state dict:\n{}'.format(state_queue_dict)) 

        # convert state dict into matrix
        state = []
        for ep in state_queue_dict.keys():
            queues = state_queue_dict[ep]
            for queue in queues.keys():
                flows = state_queue_dict[ep][queue]['queued_flows']
                for flow in flows:
                    state.append(flow)
        state = np.asarray(state)
        #print('State matrix:\n{}'.format(state))
        #print('Size of state matrix: {}'.format(state.shape))

        return state 


    def get_agent_action(self, state):
        pass




if __name__ == '__main__':
    import graphs as g 
    from demand import Demand
    from routing import RWA
    from scheduling import SRPT, BASRPT, Agent
    from simulator import DCN
    import pickle
    import networkx as nx
    import time

    # config
    max_F = 2 # max number of flows per queue
    num_episodes = 100
    num_k_paths = 1
    slot_size = 0.2
    max_time = 100
    path_figures = 'figures/'
    
    load_demands = 'pickles/demand/10_uniform_demands.obj' # path str or None
    filehandler = open(load_demands, 'rb')
    demand = pickle.load(filehandler)
    graph = demand.Graph
    edge_dict = nx.get_edge_attributes(graph, 'channels')
    num_channels = len(list(edge_dict[list(edge_dict.keys())[0]].keys()))

    # initialise routing agent
    rwa = RWA(g.gen_channel_names(num_channels), num_k_paths)
   
    # initialise scheduling agent
    #scheduler = SRPT(graph, rwa, slot_size) 
    #scheduler = BASRPT(graph, rwa, slot_size, V=500) 
    scheduler = Agent(graph, rwa, slot_size, max_F, epsilon=0.1, gamma=1.0, alpha=0.5)

    # initialise dcn simulation environment
    env = DCN(demand, scheduler, max_F, max_time)
    
    # run simulations
    for episode in range(num_episodes):
        observation = env.reset(load_demands)
        while True:
            print('------------------------------------------------------')
            print('Time: {}'.format(env.curr_time))
            print('Observation:\n{}'.format(observation))
            action = scheduler.get_action(observation)
            print('Action:\n{}'.format(action))
            observation, reward, done, info = env.step(action)
            print('Observation:\n{}'.format(observation))
            if done:
                print('Episode finished.')
                env.get_scheduling_session_summary()
                env.print_scheduling_session_summary()
                env.plot_all_queue_evolution(path_figures+'scheduler/')
                break





