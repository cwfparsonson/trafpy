from trafpy.generator.src import networks
from trafpy.generator.src import tools

import gym
import tensorflow as tf
import json
import numpy as np
import copy
import pickle
import bz2
import networkx as nx
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.animation as animation
import cv2
import io
import time


class DCN(gym.Env):

    def __init__(self, 
                 Network, 
                 Demand, 
                 Scheduler, 
                 slot_size, 
                 sim_name='dcn_sim',
                 max_flows=None, 
                 max_time=None):

        # initialise DCN environment characteristics
        self.network = Network
        self.demand = Demand
        self.scheduler = Scheduler
        self.slot_size = slot_size
        self.sim_name = sim_name 
        self.max_flows = max_flows # max number of flows per queue
        self.max_time = max_time

        # init representation generator
        self.repgen = RepresentationGenerator(self)

        # gym env reqs
        self.action_space = gym.spaces.Discrete(self.repgen.num_actions)
        self.observation_space = gym.spaces.Dict({'action_mask': gym.spaces.Box(0, 1, shape=(self.repgen.num_actions,)),
                                                  'avail_actions': gym.spaces.Box(-1, 1e12, shape=(self.repgen.num_actions, self.repgen.action_embedding_size)),
                                                  'machine_readable_network': gym.spaces.Box(-1, 1e12, shape=(self.repgen.num_actions, self.repgen.action_embedding_size))})


    def reset(self, pickled_demand_path=None, return_obs=True):
        '''
        Resets DCN simulation environment
        '''
        print('Resetting simulation \'{}\''.format(self.sim_name))

        self.curr_step = 0
        self.curr_time = 0

        if pickled_demand_path is not None:
            # load a previously saved demand object
            filehandler = open(pickled_demand_path, 'rb')
            self.demand = pickle.load(filehandler)
        self.slots_dict = self.demand.get_slots_dict(self.slot_size)
        self.check_if_pairs_valid(self.slots_dict)

        self.num_endpoints = int(len(self.network.graph['endpoints']))

        self.net_node_positions = networks.init_network_node_positions(copy.deepcopy(self.network))
        self.animation_images = []

        self.action = {'chosen_flows': []} # init

        self.channel_names = self.network.graph['channel_names'] 

        if self.demand.job_centric:
            self.network.graph['queued_jobs'] = [] # init list of curr queued jobs in network
            self.arrived_jobs = {} # use dict so can hash to make searching faster
            self.arrived_job_dicts = []
            self.completed_jobs = []
            self.dropped_jobs = []
            self.arrived_control_deps = [] # list of control dependencies
            self.arrived_control_deps_that_were_flows = []
            self.running_ops = {}
        else:
            pass
        self.connected_flows = []
        self.arrived_flows = {}
        self.arrived_flow_dicts = [] # use for calculating throughput at end
        self.completed_flows = []
        self.dropped_flows = []


        self.network = self.init_virtual_queues(self.network)
        self.queue_evolution_dict = self.init_queue_evolution(self.network)

        if return_obs:
            return self.next_observation()
        else:
            return None

    def check_if_pairs_valid(self, slots_dict):
        '''
        Since the network and the demand for a simulation are created separately,
        an easy mistake to fall into is to name the network nodes in the network
        differently from the src-dst pairs in the demand. This can lead to 
        infinite loops since the flows never get added to appropriate queues!
        This function loops through all the src-dst pairs in the first slot
        of the slots dict to try to catch this error before the simulation is
        ran
        '''
        slot = slots_dict[0]
        for event in slot['new_event_dicts']:
            if self.demand.job_centric:
                for f in event['flow_dicts']:
                    if f['src'] not in self.network.nodes or f['dst'] not in self.network.nodes:
                        sys.exit('ERROR: Demand src-dst pair names (e.g. {}-{}) different from \
                        network node names (e.g. {}). Rename one or the other to avoid errors!'.format(f['src'],f['dst'],list(self.network.nodes)[0]))
            else:
                if event['src'] not in self.network.nodes or event['dst'] not in self.network.nodes:
                    sys.exit('ERROR: Demand src-dst pair names (e.g. {}) different from \
                        network node names (e.g. {}). Rename one or the other to avoid errors!'.format(event['src'],list(self.network.nodes)[0]))

    
    def init_queue_evolution(self, Graph):
        q_dict = {src: 
                    {dst: 
                        {'times': [0],
                         'queue_lengths_info_units': [0],
                         'queue_lengths_num_flows': [0]}
                        for dst in [dst for dst in Graph.graph['endpoints'] if dst != src]}
                    for src in Graph.graph['endpoints']} 

        return q_dict
    
    def calc_queue_length(self, src, dst):
        '''
        Calc queue length in bytes at a given src-dst queue
        '''
        queue = self.network.nodes[src][dst]
        num_flows = len(queue['queued_flows'])
        
        queue_length_bytes = 0
        for flow_idx in range(num_flows):
            flow_dict = queue['queued_flows'][flow_idx]
            if flow_dict['packets'] is None:
                # scheduler agent not yet chosen this flow therefore don't 
                # know chosen packet sizes, so size == original flow size
                queued_flow_bytes = flow_dict['size']
            else:
                # scheduler agent has since chosen flow, use packets left
                # to get queue length
                queued_flow_bytes = sum(flow_dict['packets'])
            queue_length_bytes += queued_flow_bytes

        return queue_length_bytes, num_flows
    
    def update_queue_evolution(self):
        q_dict = self.queue_evolution_dict
        time = self.curr_time
        
        for src in self.network.graph['endpoints']:
            for dst in self.network.graph['endpoints']:
                if dst != src:
                    queue_length_bytes, queue_length_flows = self.calc_queue_length(src, dst)
                    q_dict[src][dst]['times'].append(time)
                    q_dict[src][dst]['queue_lengths_info_units'].append(queue_length_bytes)
                    q_dict[src][dst]['queue_lengths_num_flows'].append(queue_length_flows)
                else:
                    # can't have src == dst
                    pass
    
    def init_virtual_queues(self, Graph):
        queues_per_ep = self.num_endpoints-1
      
        # initialise queues at each endpoint as node attributes
        attrs = {ep: 
                    {dst: 
                          {'queued_flows': [],
                           'completion_times': []}
                          for dst in [dst for dst in Graph.graph['endpoints'] if dst != ep]} 
                    for ep in Graph.graph['endpoints']}

        # add these queues/attributes to endpoint nodes in graph
        nx.set_node_attributes(Graph, attrs)

        return Graph
    
    def add_flow_to_queue(self, flow_dict):
        '''
        Adds a new flow to the appropriate src-dst virtual queue in the 
        simulator's network. Also updates arrived flows record
        '''
        # add to arrived flows list
        self.register_arrived_flow(flow_dict)
        
        src = flow_dict['src']
        dst = flow_dict['dst']
        
        # check if enough space to add flow to queue
        if self.max_flows is None:
            # no limit on number of flows in queue
            add_flow = True
        else:
            # check if adding flow would exceed queue limit
            curr_num_flows = len(self.network.nodes[src][dst]['queued_flows'])
            if curr_num_flows == self.max_flows:
                # queue full, cannot add flow to queue
                add_flow = False
            elif curr_num_flows < self.max_flows:
                # there is enough space to add flow to queue
                add_flow = True
            else:
                raise Exception('Error: max flows per queue is {}, but have {} flows in queue.'.format(self.max_flows, curr_num_flows))
        
        if add_flow:
            # enough space in queue, add flow
            self.network.nodes[src][dst]['queued_flows'].append(flow_dict)
            self.network.nodes[src][dst]['completion_times'].append(None)
        else:
            # no space in queue, must drop flow
            self.dropped_flows.append(flow_dict)
            if self.demand.job_centric:
                for job_dict in self.network.graph['queued_jobs']:
                    if job_dict['job_id'] == flow_dict['job_id']:
                        # drop job
                        self.dropped_jobs.append(job_dict)
                        self.remove_job_from_queue(job_dict)
                        break
                        

    def add_job_to_queue(self, job_dict, print_times=False):
        '''
        Adds a new job with its respective flows to the appropriate
        src-dst virtual queue in the simulator's network. Aslo updates
        arrived flows record
        '''
        time_started_adding = time.time()
        # add to arrived jobs list
        self.arrived_jobs[job_dict['job_id']] = 'present' # record arrived job as being present in queue
        self.arrived_job_dicts.append(job_dict)
        self.network.graph['queued_jobs'].append(job_dict)
   
        # to ensure all child flows of completed 'flows' are updated, need to wait
        # until have gone through and queued all flows in new job graph to update
        # any flows that are completed immediately (e.g. ctrl deps coming from source)
        flows_to_complete = []
        
        start = time.time()
        for flow_dict in job_dict['flow_dicts']:
            if int(flow_dict['parent_op_run_time']) == 0:
                # parent op instantly completed
                flow_dict['time_parent_op_started'] = 0
            else:
                pass
            if self.arrived_jobs[job_dict['job_id']] == 'present':
                #if flow_dict['src'] == flow_dict['dst']:
                #    # src == dst therefore never becomes a flow
                #    flow_dict['can_schedule'] = 1 # need to change, see bottom of this method Note
                #    flow_dict['time_arrived'] = self.curr_time
                #if flow_dict['src'] != flow_dict['dst'] and int(flow_dict['size']) == 0:
                if int(flow_dict['size']) == 0:
                    # is a control dependency or src==dst therefore never becomes a flow therefore treat as control dep
                    self.arrived_control_deps.append(flow_dict)
                    if flow_dict['src'] == flow_dict['dst'] and flow_dict['dependency_type'] == 'data_dep':
                        self.arrived_control_deps_that_were_flows.append(flow_dict)
                    if int(flow_dict['parent_op_run_time']) == 0 and len(flow_dict['completed_parent_deps']) == len(flow_dict['parent_deps']):
                        # control dependency satisfied immediately
                        flow_dict['time_arrived'] = self.curr_time
                        flow_dict['can_schedule'] = 1
                        flows_to_complete.append(flow_dict)
                elif len(flow_dict['parent_deps']) == 0 and flow_dict['src'] != flow_dict['dst'] and flow_dict['size'] != 0:
                    # flow with no parent dependencies, count as having arrived immediately
                    flow_dict['time_arrived'] = job_dict['time_arrived']
                    self.add_flow_to_queue(flow_dict)
                else:
                    # flow with parent dependencies, dont count as having arrived immediately
                    self.add_flow_to_queue(flow_dict)
            else:
                # job was dropped due to full flow queue
                break
        end = time.time()
        if print_times:
            print('Time to add job to queue: {}'.format(end-start))
        
        # go back through and register any immediately completed 'flows'
        start = time.time()
        for f in flows_to_complete:
            self.register_completed_flow(f, print_times=False)
        end = time.time()
        if print_times:
            print('Time to register immediately completed flows: {}'.format(end-start))

        time_finished_adding = time.time()
        if print_times:
            print('Total time to add job to queue & register completed flows: {}'.format(time_finished_adding-time_started_adding))
       
            
    def update_curr_time(self, slot_dict):
        '''
        Updates current time of simulator using slot dict
        '''
        # update current time
        if slot_dict['ub_time'] > self.curr_time:
            # observation has a new up-to-date current time
            self.curr_time = slot_dict['ub_time']
        else:
            # observation does not have an up-to-date time
            self.curr_time += self.slot_size
            num_decimals = str(self.slot_size)[::-1].find('.')
            self.curr_time = round(self.curr_time,num_decimals)
    
    def update_running_op_dependencies(self, observation):
        '''
        Takes observation of current time slot and updates dependencies of any
        ops that are running
        '''
        # go through queued flows
        eps = self.network.graph['endpoints']
        for ep in eps:
            ep_queues = self.network.nodes[ep]
            for ep_queue in ep_queues.values():
                for flow_dict in ep_queue['queued_flows']:
                    if flow_dict['time_parent_op_started'] is not None:
                        if self.curr_time >= flow_dict['time_parent_op_started'] + flow_dict['parent_op_run_time']:
                            # parent op has finished running, can schedule flow
                            #op_id = flow_dict['job_id']+'_op_'+str(flow_dict['parent_op'])
                            op_id = flow_dict['job_id']+'_'+flow_dict['parent_op']
                            try:
                                del self.running_ops[op_id]
                            except KeyError:
                                # op has already previously been registered as completed
                                pass
                            if flow_dict['can_schedule'] == 0:
                                flow_dict['can_schedule'] = 1
                                flow_dict['time_arrived'] = self.curr_time
                                self.register_arrived_flow(flow_dict)
                            else:
                                # already registered as arrived
                                pass
                        else:
                            # child op not yet finished, cannot schedule
                            pass
                    else:
                        # can already schedule or child op not started therefore dont need to consider
                        pass

        # go through queued control dependencies
        for dep in self.arrived_control_deps:
            if dep['time_parent_op_started'] is not None and dep['time_completed'] is None:
                # dep child op has begun and dep has not been registered as completed
                if self.curr_time >= dep['time_parent_op_started'] + dep['parent_op_run_time']:
                    # child op has finished running, dependency has been completed
                    #op_id = flow_dict['job_id']+'_op_'+str(flow_dict['parent_op'])
                    op_id = dep['job_id']+'_'+dep['parent_op']
                    try:
                        del self.running_ops[op_id]
                    except KeyError:
                        # op has already previously been registered as completed
                        pass
                    dep['time_completed'] = self.curr_time
                    dep['can_schedule'] = 1
                    self.register_completed_flow(dep)
                else:
                    # parent op not yet finished
                    pass
            else:
                # parent op not yet started
                pass

        observation['network'] = self.network

        return observation

                        
    def add_flows_to_queues(self, observation):
        '''
        Takes observation of current time slot and updates virtual queues in
        network
        '''
        slot_dict = observation['slot_dict']

        if len(slot_dict['new_event_dicts']) == 0:
            # no new event(s)
            pass
        else:
            # new event(s)
            num_events = int(len(slot_dict['new_event_dicts']))
            for event in range(num_events):
                event_dict =    slot_dict['new_event_dicts'][event]
                if event_dict['establish'] == 0:
                    # event is a take down event, don't need to consider
                    pass
                elif event_dict['establish'] == 1:
                    if self.demand.job_centric:
                        self.add_job_to_queue(event_dict, print_times=False)
                    else:
                        self.add_flow_to_queue(event_dict)
        
        observation['network'] = self.network # update osbervation's network
        
        return observation
    
    def remove_job_from_queue(self, job_dict):
        idx = 0
        queued_jobs = copy.deepcopy(self.network.graph['queued_jobs'])

        eps = copy.deepcopy(self.network.graph['endpoints'])
        for ep in eps:
            ep_queues = self.network.nodes[ep]
            for ep_queue in ep_queues.values():
                for f in ep_queue['queued_flows']:
                    if f['job_id'] == job_dict['job_id']:
                        self.remove_flow_from_queue(f)
                    else:
                        # flow does not belong to job being removed
                        pass

        self.arrived_jobs[job_dict['job_id']] = 'removed' 
        
    
    def remove_flow_from_queue(self, flow_dict):
        if flow_dict['src'] == flow_dict['dst']:
            pass
        else:
            sn = flow_dict['src']
            dn = flow_dict['dst']
            queued_flows = self.network.nodes[sn][dn]['queued_flows']
            idx = self.find_flow_idx(flow_dict, queued_flows)
            del self.network.nodes[sn][dn]['queued_flows'][idx]
            del self.network.nodes[sn][dn]['completion_times'][idx]
    
    def register_completed_flow(self, flow_dict, print_times=False):
        '''
        Takes a completed flow, appends it to list of completed flows, records
        time at which it was completed, and removes it from queue. If 'flow' in
        fact never become a flow (i.e. had src == dst or was control dependency
        with size == 0), will update dependencies but won't append to completed
        flows etc.
        '''
      
        # record time at which flow was completed
        start = time.time()
        flow_dict['time_completed'] = copy.copy(self.curr_time)
        if flow_dict['size'] != 0 and flow_dict['src'] != flow_dict['dst']:
            # flow was an actual flow
            self.completed_flows.append(flow_dict)
            fct = flow_dict['time_completed'] - flow_dict['time_arrived']
        else:
            # 'flow' never actually became a flow (src == dst or control dependency)
            pass
        end = time.time()
        if print_times:
            print('\nTime to record time flow completed: {}'.format(end-start))
        
        start = time.time()
        f = copy.copy(flow_dict)
        if flow_dict['size'] != 0:
            # remove flow from queue
            self.remove_flow_from_queue(flow_dict)
        else:
            # never became flow 
            pass
        end = time.time()
        if print_times:
            print('Time to remove flow from global queue: {}'.format(end-start))
        
        start = time.time()
        if self.demand.job_centric:
            # make any necessary job completion & job dependency changes
            self.update_completed_flow_job(f)
        end = time.time()
        if print_times:
            print('Time to record any job completions and job dependency changes: {}'.format(end-start))


    def register_arrived_flow(self, flow_dict):
        # register
        if flow_dict['can_schedule'] == 1:
            # flow is ready to be scheduled therefore can count as arrived
            if self.demand.job_centric:
                arrival_id = flow_dict['job_id']+'_'+flow_dict['flow_id']
            else:
                arrival_id = flow_dict['flow_id']
            try:
                _ = self.arrived_flows[arrival_id]
                # flow already counted as arrived
            except KeyError:
                # flow not yet counted as arrived
                if flow_dict['src'] != flow_dict['dst'] and flow_dict['size'] != 0:
                    if flow_dict['time_arrived'] is None:
                        # record time flow arrived
                        flow_dict['time_arrived'] = self.curr_time
                    else:
                        # already recorded time of arrival
                        pass
                    self.arrived_flows[arrival_id] = 'present'
                    self.arrived_flow_dicts.append(flow_dict)
                else:
                    # 'flow' never actually becomes flow (is ctrl dependency or src==dst)
                    pass
        else:
            # can't yet schedule therefore don't count as arrived
            pass
        


    def update_flow_packets(self, flow_dict):
        '''
        Takes flow dict that has been schedueled to be activated for curr
        time slot and removes corresponding number of packets flow in queue
        '''
        packet_size = flow_dict['packets'][0] 
        path_links = self.get_path_edges(flow_dict['path'])
        link_bws = []
        for link in path_links:
            link_bws.append(self.network[link[0]][link[1]]['max_channel_capacity'])
        lowest_bw = min(link_bws)
        size_per_slot = lowest_bw/(1/self.slot_size)
        packets_per_slot = int(size_per_slot / packet_size) # round down 
        
        sn = flow_dict['src']
        dn = flow_dict['dst']
        queued_flows = self.network.nodes[sn][dn]['queued_flows']
        idx = self.find_flow_idx(flow_dict, queued_flows)
        queued_flows[idx]['packets'] = queued_flows[idx]['packets'][packets_per_slot:]
        
        updated_flow = copy.copy(queued_flows[idx])
        if len(updated_flow['packets']) == 0:
            # all packets transported, flow completed
            self.register_completed_flow(updated_flow)
                
        
        return flow_dict

    def get_current_queue_states(self):
        '''
        Returns list of all queues in network
        '''
        queues = []
        eps = self.network.graph['endpoints']
        for ep in eps:
            ep_queues = self.network.nodes[ep]
            for ep_queue in ep_queues.values():
                if len(ep_queue['queued_flows'])!=0:
                    queues.append(ep_queue)
        return queues
                

    def next_observation(self):
        '''
        Compiles simulator data and returns observation
        '''
        try:
            observation = {'slot_dict': self.slots_dict[self.curr_step],
                           'network': copy.deepcopy(self.network)}
            self.update_curr_time(observation['slot_dict'])
            # add any new events (flows or jobs) to queues
            observation = self.add_flows_to_queues(observation)
        except KeyError:
            # curr step exceeded slots dict indices, no new flows/jobs arriving
            slot_keys = list(self.slots_dict.keys())
            observation = {'slot_dict': self.slots_dict[slot_keys[-1]],
                           'network': copy.deepcopy(self.network)}
            self.update_curr_time(observation['slot_dict'])
        
        # update any dependencies of running ops
        if self.demand.job_centric:
            observation = self.update_running_op_dependencies(observation)

        # create machine readable version of current network state
        observation['machine_readable_network'] = self.repgen.gen_machine_readable_network_observation(observation['network'])

        # update available actions
        observation['avail_actions'], action_dict = observation['machine_readable_network']

        # mask any placeholder actions not occupied by flows
        action_mask = [0 for _ in range(self.repgen.num_actions)]
        action_indices_with_flow_present = self.get_indices_of_action_placeholders_occupied_by_flows(action_dict)
        for i in action_indices_with_flow_present:
            action_mask[i] = 1
        observation['action_mask'] = action_mask

            
        return observation

    def get_indices_of_action_placeholders_occupied_by_flows(self, action_dict):
        indices = []
        for index in action_dict.keys():
            if action_dict[index]['flow_present'] == 1:
                # flow present
                indices.append(index)
            else:
                # no flow present at this placeholder action
                pass

        return indices
                



    def check_num_channels_used(self, graph, edge):
        '''
        Checks number of channels currently in use on given edge in graph
        '''
        num_channels = graph.graph['num_channels_per_link']
        num_channels_used = 0

        for channel in self.channel_names:
            channel_used = self.check_if_channel_used(graph, [edge], channel)
            if channel_used:
                num_channels_used += 1
            else:
                pass
        
        return num_channels_used, num_channels


    def draw_network_state(self,
                           draw_flows=True,
                           draw_ops=True,
                           draw_node_labels=False,
                           ep_label='server',
                           appended_node_size=300, 
                           network_node_size=2000,
                           appended_node_x_spacing=5,
                           appended_node_y_spacing=0.75,
                           font_size=15, 
                           linewidths=1, 
                           fig_scale=2):
        '''
        Draws network state as matplotlib figure
        '''
    
        network = copy.deepcopy(self.network)

        fig = plt.figure(figsize=[15*fig_scale,15*fig_scale])

        # add nodes and edges
        pos = {}
        flows = []
        network_nodes = []
        ops = []
        network_nodes_dict = networks.get_node_type_dict(network, self.network.graph['node_labels'])
        for nodes in list(network_nodes_dict.values()):
            for network_node in nodes:
                pos[network_node] = self.net_node_positions[network_node]

        eps = network.graph['endpoints']
        for ep in eps:
            ep_queues = network.nodes[ep]
            y_offset = -appended_node_y_spacing
            for ep_queue in ep_queues.values():
                for flow in ep_queue['queued_flows']:
                    if self.demand.job_centric:
                        f_id = str(flow['job_id']+'_'+flow['flow_id'])
                    else:
                        f_id = str(flow['flow_id'])
                    network.add_node(f_id)
                    network.add_edge(f_id, flow['src'], type='queue_link')
                    flows.append(f_id)
                    pos[f_id] = (list(pos[flow['src']])[0]+appended_node_x_spacing, list(pos[flow['src']])[1]+y_offset)
                    y_offset-=appended_node_y_spacing

        if self.demand.job_centric:
            for ep in eps:
                y_offset = -appended_node_y_spacing
                for op in self.running_ops.keys():
                    op_machine = self.running_ops[op]
                    if ep == op_machine:
                        network.add_node(op)
                        network.add_edge(op, op_machine, type='op_link')
                        ops.append(op)
                        pos[op] = (list(pos[op_machine])[0]-appended_node_x_spacing, list(pos[op_machine])[1]+y_offset)
                        y_offset-=appended_node_y_spacing
                    else:
                        # op not placed on this end point machine
                        pass
        else:
            pass

        # find edges
        fibre_links = []
        queue_links = []
        op_links = []
        for edge in network.edges:
            if 'channels' in network[edge[0]][edge[1]].keys():
                # edge is a fibre link
                fibre_links.append(edge)
            elif network[edge[0]][edge[1]]['type'] == 'queue_link':
                # edge is a queue link
                queue_links.append(edge)
            elif network[edge[0]][edge[1]]['type'] == 'op_link':
                # edge is a queue link
                op_links.append(edge)
            else:
                sys.exit('Link type not recognised.')

        # network nodes
        node_colours = iter(['#25c44d', '#36a0c7', '#e8b017', '#6115a3', '#160e63']) # server, rack, edge, agg, core
        for node_type in self.network.graph['node_labels']:
            nx.draw_networkx_nodes(network, 
                                   pos, 
                                   nodelist=network_nodes_dict[node_type],
                                   node_size=network_node_size, 
                                   node_color=next(node_colours), 
                                   linewidths=linewidths, 
                                   label=node_type)

        if draw_flows:
            # flows
            nx.draw_networkx_nodes(network, 
                                   pos, 
                                   nodelist=flows,
                                   node_size=appended_node_size, 
                                   node_color='#bd3937', 
                                   linewidths=linewidths, 
                                   label='Queued flow')
            # queue links
            nx.draw_networkx_edges(network, 
                                   pos,
                                   edgelist=queue_links,
                                   edge_color='#bd3937',
                                   alpha=0.5,
                                   width=0.5,
                                   style='dashed',
                                   label='Queue link')

        if draw_ops and self.demand.job_centric:
            # ops
            nx.draw_networkx_nodes(network, 
                                   pos, 
                                   nodelist=ops,
                                   node_size=appended_node_size, 
                                   node_color='#1e9116', 
                                   linewidths=linewidths, 
                                   label='Running op')
            # op links
            nx.draw_networkx_edges(network, 
                                   pos,
                                   edgelist=op_links,
                                   edge_color='#1e9116',
                                   alpha=0.5,
                                   width=0.5,
                                   style='dashed',
                                   label='Op link')




        # fibre links
        nx.draw_networkx_edges(network, 
                               pos,
                               edgelist=fibre_links,
                               edge_color='k',
                               width=3,
                               label='Fibre link')

        if draw_node_labels:
            # nodes
            nx.draw_networkx_labels(network, 
                                    pos, 
                                    font_size=font_size, 
                                    font_color='k', 
                                    font_family='sans-serif', 
                                    font_weight='normal', 
                                    alpha=1.0)

        plt.legend(labelspacing=2.5)
    
        return fig, network, pos


    def render_network(self, action=None, fig_scale=1):
        '''
        Renders network state as matplotlib figure and, if specified, renders
        chosen action(s) (lightpaths) on top of figure
        '''
        fig, network, pos = self.draw_network_state(draw_flows=True,
                                                    draw_ops=True, 
                                                    fig_scale=fig_scale)
        
        if action is not None:
            # init fibre link labels
            fibre_link_labels = {}
            for edge in network.edges:
                if 'flow' in edge[0] or 'flow' in edge[1] or '_op_' in edge[0] or '_op_' in edge[1]:
                    # edge is not a fibre, dont need to label
                    pass
                else:
                    # fibre not yet added
                    fibre_link_labels[(edge[0], edge[1])] = 0

            # render selected actions/chosen lightpaths in network
            active_lightpath_edges = []
            for flow in action['chosen_flows']:
                path_edges = self.get_path_edges(flow['path'])
                f_id = str(flow['job_id']+'_'+flow['flow_id'])
                queue_link = [f_id, flow['src']]
                path_edges.append(queue_link)
                for edge in path_edges:  
                    active_lightpath_edges.append(edge)
                    if '_flow_' in edge[0] or '_flow_' in edge[1] or '_op_' in edge[0] or '_op_' in edge[1]:
                        # edge is not a fibre, dont need to label
                        pass
                    else:
                        # edge is fibre, label with number of active lightpaths
                        try:
                            fibre_link_labels[(edge[0], edge[1])] += 1
                        except KeyError:
                            fibre_link_labels[(edge[1], edge[0])] += 1

            # format fibre link labels
            for link in fibre_link_labels.keys():
                num_channels_used = fibre_link_labels[link]
                fibre_label = '{}/{}'.format(str(num_channels_used),self.network.graph['num_channels_per_link']) 
                fibre_link_labels[link] = fibre_label

            # lightpaths
            nx.draw_networkx_edges(network, 
                                   pos,
                                   edgelist=active_lightpath_edges,
                                   edge_color='#e80e0e',
                                   alpha=0.5,
                                   width=15,
                                   label='Active lightpath')
            
            # lightpath labels
            nx.draw_networkx_edge_labels(network,
                                         pos,
                                         edge_labels=fibre_link_labels,
                                         font_size=15,
                                         font_color='k',
                                         font_family='sans-serif',
                                         font_weigh='normal',
                                         alpha=1.0)


        else:
            # no action given, just render network queue state(s)
            pass

        plt.title('Time: {}'.format(self.curr_time), fontdict={'fontsize': 100})
        
        return fig



    def conv_fig_to_image(self, fig):
        '''
        Takes matplotlib figure and converts into numpy array of RGB pixel values
        '''
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=self.dpi)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr,1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    
    def render(self, action=None, dpi=300, fig_scale=1):
        '''
        Renders current network state for final animation at end of scheduling
        session. If action is None, will only render network queue state(s)
        rather than also rendering selected lightpaths.
        '''
        self.dpi = dpi
        fig = self.render_network(copy.deepcopy(action), fig_scale=fig_scale)
        self.animation_images.append(self.conv_fig_to_image(fig))
        plt.close() 
    
    def register_completed_job(self, job_dict):
        
        # record time at which job was completed
        job_dict['time_completed'] = copy.copy(self.curr_time)
        self.completed_jobs.append(job_dict)
        jct = job_dict['time_completed']-job_dict['time_arrived']
        
        # remove job from queue
        self.remove_job_from_queue(job_dict) 
    
    def update_completed_flow_job(self, completed_flow):
        '''
        Updates dependencies of other flows in job of completed flow.
        Checks if all flows in job of completed flow have been completed.
        If so, will update stat trackers & remove from tracking list
        '''
        # update any dependencies of flow in job
        self.update_job_flow_dependencies(completed_flow)

        # check if job completed
        job_id = completed_flow['job_id']
        job_flows_completed, job_ctrl_deps_completed = True, True
        eps = self.network.graph['endpoints']
        # check job flows
        for ep in eps:
            ep_queues = self.network.nodes[ep]
            for ep_queue in ep_queues.values():
                for f in ep_queue['queued_flows']:
                    if f['job_id'] == job_id and f['time_completed'] is None:
                        # job still has at least one uncompleted flow left
                        job_flows_completed = False
                        break

        # check job ctrl deps
        for dep in self.arrived_control_deps:
            if dep['job_id'] == job_id and dep['time_completed'] is None:
                # job still has at least one uncompleted control dependency left
                job_ctrl_deps_completed = False
                break

        if job_flows_completed == True and job_ctrl_deps_completed == True:
            # register completed job
            for job in self.network.graph['queued_jobs']:
                if job['job_id'] == job_id:
                    self.register_completed_job(job)
                    break
        else:
            # job not yet completed, has outstanding dependencies
            pass
            
    
    def update_job_flow_dependencies(self, completed_flow):
        '''
        Go through flows in job and update any flows that were waiting for 
        completed flow to arrive before being able to be scheduled
        '''
        completed_child_dependencies = completed_flow['child_deps']
        eps = self.network.graph['endpoints']
        
        for child_dep in completed_child_dependencies:
            # go through queued flows
            for ep in eps:
                ep_queues = self.network.nodes[ep]
                for ep_queue in ep_queues.values():
                    for flow_dict in ep_queue['queued_flows']:
                        if flow_dict['job_id'] == completed_flow['job_id']:
                            # only update if part of same job!
                            if flow_dict['flow_id'] == child_dep:
                                # child dependency found
                                flow_dict['completed_parent_deps'].append(completed_flow['flow_id'])
                                if len(flow_dict['completed_parent_deps']) == len(flow_dict['parent_deps']):
                                    # parent dependencies of child op have been completed
                                    if flow_dict['parent_op_run_time'] > 0 and flow_dict['time_parent_op_started'] == None:
                                        # child op of flow has non-zero run time and has not yet started
                                        flow_dict['time_parent_op_started'] = self.curr_time
                                        #op_id = flow_dict['job_id']+'_op_'+str(flow_dict['parent_op'])
                                        op_id = flow_dict['job_id']+'_'+flow_dict['parent_op']
                                        op_machine = flow_dict['src']
                                        self.running_ops[op_id] = op_machine
                                    else:
                                        # child op of flow has 0 run time, can schedule flow now
                                        flow_dict['can_schedule'] = 1
                                        flow_dict['time_arrived'] = self.curr_time
                                        self.register_arrived_flow(flow_dict)
                                else:
                                    # still can't schedule
                                    pass
                            else:
                                # flow is not a child dep of completed flow
                                pass
                        else:
                            # flow not part of same job as completed flow
                            pass
                          
            for child_dep in completed_child_dependencies:
                # go through arrived control dependencies
                for control_dep in self.arrived_control_deps:
                    if control_dep['job_id'] == completed_flow['job_id']:
                        # only update if part of same job!
                        if control_dep['flow_id'] == child_dep:
                            # child dependency found
                            control_dep['completed_parent_deps'].append(completed_flow['flow_id'])
                            if len(control_dep['completed_parent_deps']) == len(control_dep['parent_deps']):
                                # parent dependencies of child op have been completed
                                if control_dep['parent_op_run_time'] > 0 and control_dep['time_parent_op_started'] == None:
                                    # child op of control dep has non-zero run time and has not yet started
                                    control_dep['time_parent_op_started'] = self.curr_time
                                    control_dep['time_arrived'] = self.curr_time
                                    #op_id = control_dep['job_id']+'_op_'+str(flow_dict['parent_op'])
                                    op_id = control_dep['job_id']+'_'+control_dep['parent_op']
                                    op_machine = control_dep['src']
                                    self.running_ops[op_id] = op_machine
                                else:
                                    # child op of control dep has 0 run time, control dependency has been satisfied
                                    control_dep['can_schedule'] = 1
                                    control_dep['time_arrived'] = self.curr_time
                            else:
                                # still can't schedule
                                pass
                        else:
                            # dep is not a child dep of completed flow
                            pass
                    else:
                        # dep not part of same job as completed flow
                        pass
                    

    def update_flow_attrs(self, chosen_flows):
        for flow in chosen_flows:
            sn = flow['src']
            dn = flow['dst']
            queued_flows = self.network.nodes[sn][dn]['queued_flows']
            idx = self.find_flow_idx(flow, queued_flows)
            dated_flow = self.network.nodes[sn][dn]['queued_flows'][idx]
            if dated_flow['packets'] is None:
                # udpate flow packets and k shortest paths
                dated_flow['packets'] = flow['packets']
                dated_flow['k_shortest_paths'] = flow['k_shortest_paths']
            else:
                # agent updates already applied
                pass

    def take_action(self, action):
        # unpack chosen action
        chosen_flows = action['chosen_flows']
        
        # update any flow attrs in dcn network that have been updated by agent
        self.update_flow_attrs(chosen_flows)

        # establish chosen flows
        for chosen_flow in chosen_flows:
            #flow_established, _ = self.check_flow_present(chosen_flow, self.connected_flows)
            if len(chosen_flows) == 0:
                # no flows chosen
                pass
            #elif flow_established:
            elif self.check_flow_present(chosen_flow, self.connected_flows):
                # chosen flow already established, leave
                pass
            else:
                # chosen flow not already established, establish connection
                self.set_up_connection(chosen_flow)

        # take down replaced flows
        for prev_chosen_flow in self.connected_flows:
            #flow_established, _ = self.check_flow_present(prev_chosen_flow, chosen_flows)
            if self.connected_flows == 0:
                # no flows chosen previously
                pass
            #elif flow_established:
            elif self.check_flow_present(prev_chosen_flow, chosen_flows):
                # prev chosen flow has been re-chosen, leave
                pass
            else:
                # prev chosen flow not re-chosen, take down connection
                self.take_down_connection(prev_chosen_flow)

        #self.update_completed_flows(chosen_flows) 
        for flow in chosen_flows:
            self.update_flow_packets(flow)
        
        
        #self.update_completed_flows(chosen_flows) 

        self.update_queue_evolution()

        # all chosen flows established and any removed flows taken down
        # update connected_flows
        self.connected_flows = chosen_flows.copy()


    def step(self, action):
        '''
        Performs an action in the DCN simulation environment, moving simulator
        to next step
        '''
        self.action = action # save action
        
        self.take_action(action)

        self.curr_step += 1
        reward = self.calc_reward()
        done = self.check_if_done()
        info = None
        obs = self.next_observation()



        # # DEBUG:
        # queues = self.get_current_queue_states()
        # print('arrived flows:\n{}'.format(self.arrived_flow_dicts))
        # print('arrived flow ids:\n{}'.format(self.arrived_flows))
        # print('Num completed flows: {}/{}'.format(len(self.completed_flows), self.demand.num_flows))
        # print('Queues being given to scheduler:')
        # i = 0
        # for q in queues:
           # print('queue {}:\n{}'.format(i, q))
           # i+=1
        # print('Incomplete control deps:')
        # for c in self.arrived_control_deps:
           # if c['time_completed'] is None or c['time_parent_op_started'] is None:
               # print(c)
           # else:
               # pass

        return obs, reward, done, info
  
    
    def calc_num_queued_flows_num_full_queues(self):
        '''
        Calc num queued flows and full queues in network
        '''
        num_full_queues = 0
        num_queued_flows = 0
        
        eps = self.network.graph['endpoints']
        for ep in eps:
            ep_queues = self.network.nodes[ep]
            for ep_queue in ep_queues.values():
                num_flows_in_queue = len(ep_queue['queued_flows'])
                num_queued_flows += num_flows_in_queue
                if self.max_flows is not None:
                    if num_flows_in_queue == self.max_flows:
                        num_full_queues += 1
                    elif num_flows_in_queue > self.max_flows:
                        raise Exception('Error: max flows per queue is {}, but have {} flows in queue.'.format(self.max_flows, num_flows_in_queue))
                    else:
                        pass
                else:
                    # no max number of flows therefore no full queues
                    pass

        return num_queued_flows, num_full_queues


    def calc_num_queued_jobs(self):
        '''Calc num queued jobs in network.'''
        return len(self.network.graph['queued_jobs'])


    def calc_reward(self):
        if self.max_flows is None:
            # no maximum number of flows per queue therefore no full queues
            num_queued_flows, _ = self.calc_num_queued_flows_num_full_queues()
            r = - (self.slot_size * num_queued_flows)
        else:
            num_queued_flows, num_full_queues = self.calc_num_queued_flows_num_full_queues()
            r = - (self.slot_size) * (num_queued_flows + num_full_queues)

        return r
    
    def save_rendered_animation(self, path_animation, fps=1, bitrate=1800, animation_name='anim'):
        if len(self.animation_images) > 0:
            # rendered images ready to be made into animation
            print('\nSaving scheduling session animation...')
            plt.close()
            fig = plt.figure()
            fig.patch.set_visible(False)
            ax = fig.gca()
            ax.axis('off')
            plt.box(False)
            images = []
            for im in self.animation_images:
                img = plt.imshow(im)
                images.append([img])
            ani = animation.ArtistAnimation(fig,
                                            images,
                                            interval=1000,
                                            repeat_delay=5000,
                                            blit=True)
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, bitrate=bitrate) #codec=libx264
            ani.save(path_animation + animation_name + '.mp4', writer=writer, dpi=self.dpi)
            print('Animation saved.')
        else:
            print('No images were rendered during scheduling session, therefore\
                    cannot make animation.')
    

    def check_if_any_flows_arrived(self):
        if len(self.arrived_flows.keys()) == 0:
            sys.exit('Scheduling session ended, but no flows were recorded as \
                    having arrived. Consider whether the demand data you gave \
                    to the simulator actually contains non-zero sized messages \
                    which become flows.')
        else:
            pass


    def check_if_done(self):
        '''
        Checks if all flows (if flow centric) or all jobs (if job centric) have arrived &
        been completed &/or dropped
        '''
        if self.max_time is None:
            if self.demand.job_centric:
                if (len(self.arrived_jobs.keys()) != len(self.completed_jobs) and len(self.arrived_jobs.keys()) != 0 and self.demand.num_demands != len(self.dropped_jobs)+len(self.completed_jobs)) or len(self.arrived_jobs.keys()) != self.demand.num_demands:
                    return False
                else:
                    self.check_if_any_flows_arrived()
                    return True
            else:
                if (len(self.arrived_flows.keys()) != len(self.completed_flows) and len(self.arrived_flows.keys()) != 0 and self.demand.num_demands != len(self.dropped_flows)+len(self.completed_flows)) or len(self.arrived_flows.keys()) != self.demand.num_demands:
                    return False
                else:
                    self.check_if_any_flows_arrived()
                    return True

        else:
            if self.demand.job_centric:
                if self.curr_time >= self.max_time:
                    self.check_if_any_flows_arrived()
                    return True
                elif (len(self.arrived_jobs.keys()) != len(self.completed_jobs) and len(self.arrived_jobs.keys()) != 0 and self.demand.num_demands != len(self.dropped_jobs)+len(self.completed_jobs)) or len(self.arrived_jobs.keys()) != self.demand.num_demands:
                    return False
                else:
                    self.check_if_any_flows_arrived()
                    return True
            else:
                if self.curr_time >= self.max_time:
                    self.check_if_any_flows_arrived()
                    return True
                elif (len(self.arrived_flows.keys()) != len(self.completed_flows) and len(self.arrived_flows.keys()) != 0 and self.demand.num_demands != len(self.dropped_flows)+len(self.completed_flows)) or len(self.arrived_flows.keys()) != self.demand.num_demands:
                    return False
                else:
                    self.check_if_any_flows_arrived()
                    return True


    def get_path_edges(self, path):
        '''
        Takes a path and returns list of edges in the path

        Args:
        - path (list): path in which you want to find all edges

        Returns:
        - edges (list of lists): all edges contained within the path
        '''
        num_nodes = len(path)
        num_edges = num_nodes - 1
        edges = [path[edge:edge+2] for edge in range(num_edges)]

        return edges
    
    
    def check_flow_present(self, flow, flows):
        '''
        Checks if flow is present in a list of flows. Assumes the following 
        flow features are unique and unchanged properties of each flow:
        - flow size
        - source
        - destination
        - time arrived
        - flow_id
        - job_id

        Args:
        - flow (dict): flow dictionary
        - flows (list of dicts) list of flows in which to check if flow is
        present
        '''
        size = flow['size']
        src = flow['src']
        dst = flow['dst']
        time_arrived = flow['time_arrived']
        flow_id = flow['flow_id']
        job_id = flow['job_id']

        idx = 0
        for f in flows:
            if f['size']==size and f['src']==src and f['dst']==dst and f['time_arrived']==time_arrived and f['flow_id']==flow_id and f['job_id']==job_id:
                # flow found in flows
                return True, idx
            else:
                # flow not found, move to next f in flows
                idx += 1

        return False, idx
    
    def find_flow_idx(self, flow, flows):
        '''
        Finds flow idx in a list of flows. Assumes the following 
        flow features are unique and unchanged properties of each flow:
        - flow size
        - source
        - destination
        - time arrived
        - flow_id
        - job_id

        Args:
        - flow (dict): flow dictionary
        - flows (list of dicts) list of flows in which to find flow idx
        '''
        size = flow['size']
        src = flow['src']
        dst = flow['dst']
        time_arrived = flow['time_arrived']
        flow_id = flow['flow_id']
        job_id = flow['job_id']

        idx = 0
        for f in flows:
            if f['size']==size and f['src']==src and f['dst']==dst and f['time_arrived']==time_arrived and f['flow_id'] == flow_id and f['job_id'] == job_id:
                # flow found in flows
                return idx
            else:
                # flow not found, move to next f in flows
                idx += 1
        
        return sys.exit('Flow not found in list of flows')
    
    
    def check_if_channel_used(self, graph, edges, channel):
        '''
        Takes list of edges to see if any one of the edges have used a certain
        channel

        Args:
        - edges (list of lists): edges we want to check if have used certain
        channel
        - channel (label): channel we want to check if has been used by any
        of the edges

        Returns:
        - True/False
        '''
        channel_used = False
        
        num_edges = len(edges)
        for edge in range(num_edges):
            node_pair = edges[edge]
            capacity = graph[node_pair[0]][node_pair[1]]['channels'][channel]
            if round(capacity,0) != round(graph[node_pair[0]][node_pair[1]]['max_channel_capacity'],0):
                channel_used = True
                break
            else:
                continue

        return channel_used
        
    def set_up_connection(self, flow):
        '''
        Sets up connection between src-dst node pair by removing capacity from
        all edges in path connecting them. Also updates graph's global curr 
        network capacity used property
        
        Args:
        - flow (dict): flow dict containing flow info to set up
        '''
        path = flow['path']
        channel = flow['channel']
        flow_size = flow['size']

        edges = self.get_path_edges(path)

        num_edges = len(edges)
        for edge in range(num_edges):
            node_pair = edges[edge]
            # update edge property
            self.network[node_pair[0]][node_pair[1]]['channels'][channel] -= flow_size
            # update global graph property
            self.network.graph['curr_nw_capacity_used'] += flow_size
        self.network.graph['num_active_connections'] += 1
        
    

    def take_down_connection(self, flow):
        '''
        Removes established connection by adding capacity back onto all edges
        in the path connecting the src-dst node pair. Also updates graph's
        global curr network capacity used property

        Args:
        - flow (dict): flow dict containing info of flow to take down
        '''
        path = flow['path']
        channel = flow['channel']
        flow_size = flow['size']

        edges = self.get_path_edges(path)
        
        num_edges = len(edges)
        for edge in range(num_edges):
            node_pair = edges[edge]
            # update edge property
            self.network[node_pair[0]][node_pair[1]]['channels'][channel] += flow_size
            # update global graph property
            self.network.graph['curr_nw_capacity_used'] -= flow_size
        self.network.graph['num_active_connections'] -= 1

    def save_sim(self, 
                 path, 
                 name=None, 
                 overwrite=False, 
                 zip_data=True, 
                 print_times=True):
        '''
        Save self (i.e. object) using pickle
        '''
        start = time.time()
        if name is None:
            name = 'sim_jobcentric_{}'.format(str(self.demand.num_demands), str(self.demand.job_centric))
        else:
            # name already given
            pass
        
        filename = path+name+'.obj'
        if overwrite:
            # overwrite prev saved file
            pass
        else:
            # avoid overwriting
            v = 2
            while os.path.exists(str(filename)):
                filename = path+name+'_v{}'.format(v)+'.obj'
                v += 1
        if zip_data:
            filehandler = bz2.open(filename, 'wb')
        else:
            filehandler = open(filename, 'wb')
        pickle.dump(dict(self.__dict__), filehandler)
        filehandler.close()
        end = time.time()
        if print_times:
            print('Time to save sim: {}'.format(end-start))

        




class RepresentationGenerator:
    def __init__(self, env):
        self.env = env
        
        # init network params
        self.num_nodes, self.num_pairs, self.node_to_index, self.index_to_node = tools.get_network_params(env.network.graph['endpoints'])
        self.ep_label = env.network.graph['endpoint_label']
        
        # init onehot encoding
        self.onehot_endpoints, self.endpoint_to_onehot = self.onehot_encode_endpoints()
        self.onehot_paths, self.path_to_onehot = self.onehot_encode_paths()

        # get init representation to get action embedding size
        init_rep, _ = self.gen_machine_readable_network_observation(self.env.network)
        self.num_actions = tf.shape(init_rep)[0]
        self.action_embedding_size = tf.shape(init_rep)[1]
        
    def onehot_encode_endpoints(self):
        onehot_endpoints = tf.one_hot(indices=list(self.index_to_node.keys()), depth=self.num_nodes)
        endpoint_to_onehot = {endpoint: onehot for endpoint, onehot in zip(self.index_to_node.values(), onehot_endpoints)}
        
        return onehot_endpoints, endpoint_to_onehot
    
    def onehot_encode_paths(self):
        num_k_paths = self.env.scheduler.RWA.num_k
        all_paths = []
        for src in self.node_to_index.keys():
            for dst in self.node_to_index.keys():
                if src == dst:
                    pass
                else:
                    paths = self.env.scheduler.RWA.k_shortest_paths(self.env.network, src, dst)
                    for path in paths:
                        if path[::-1] not in all_paths and path not in all_paths:
                            all_paths.append(path)
        indices = [i for i in range(len(all_paths))]
        path_to_index = {json.dumps(path): index for path, index in zip(all_paths, indices)}
        index_to_path = {index: json.dumps(path) for index, path in zip(indices, all_paths)}
        onehot_paths = tf.one_hot(indices=list(index_to_path.keys()), depth=len(all_paths))
        path_to_onehot = {path: onehot for path, onehot in zip(index_to_path.values(), onehot_paths)}
        
        return onehot_paths, path_to_onehot        
    
    def gen_machine_readable_network_observation(self, network_observation, dtype=tf.float32):
        num_placeholder_flows = self.num_nodes * self.env.max_flows * (self.num_nodes - 1)
        num_actions = num_placeholder_flows * self.env.scheduler.RWA.num_k
        
        # init action representations with empty (no) flows
        action_indices = [i for i in range(num_actions)]
        action_dict = {index: {'src': tf.cast(tf.zeros(len(self.onehot_endpoints[0])), dtype=dtype),
                               'dst': tf.cast(tf.zeros(len(self.onehot_endpoints[0])), dtype=dtype),
                               'path': tf.cast(tf.zeros(len(self.onehot_paths[0])), dtype=dtype),
                               'size': tf.cast(-1, dtype=dtype),
                               'time_arrived': tf.cast(-1, dtype=dtype),
                               'selected': tf.cast(0, dtype=dtype),
                               'null_action': tf.cast(0, dtype=dtype),
                               'flow_present': tf.cast(0, dtype=dtype)} for index in action_indices}
        
        # go through network_observation queued flows and update action_dict representations
        action_iterator = iter(action_indices)
        for node in network_observation.nodes:
            if node[:len(self.ep_label)] == self.ep_label:
                queues = network_observation.nodes[node]
                for queue in queues.values():
                    for flow in queue['queued_flows']:
                        paths = self.env.scheduler.RWA.k_shortest_paths(self.env.network, flow['src'], flow['dst'])
                        for path in paths:
                            # each path is a separate action
                            idx = next(action_iterator)
                            action_dict[idx]['src'] = tf.cast(self.endpoint_to_onehot[flow['src']], dtype=dtype)
                            action_dict[idx]['dst'] = tf.cast(self.endpoint_to_onehot[flow['dst']], dtype=dtype)
                            try:
                                action_dict[idx]['path'] = tf.cast(self.path_to_onehot[json.dumps(path)], dtype=dtype)
                            except KeyError:
                                action_dict[idx]['path'] = tf.cast(self.path_to_onehot[json.dumps(path[::-1])], dtype=dtype)
                            action_dict[idx]['size'] = tf.cast(flow['size'], dtype=dtype)
                            action_dict[idx]['time_arrived'] = tf.cast(flow['time_arrived'], dtype=dtype)
                            action_dict[idx]['selected'] = tf.cast(0, dtype=dtype)
                            action_dict[idx]['null_action'] = tf.cast(0, dtype=dtype)
                            action_dict[idx]['flow_present'] = tf.cast(1, dtype=dtype)
            
            else:
                # flow placeholder in queue is empty
                pass
            
        # concat each action_dict key's values into single action_vector and stack all action vectors into single tensor
        action_stack = []
        for action in action_dict.keys():
            action_vector = self._stack_list_of_tensors([s for s in action_dict[action].values()], dtype=dtype)
            action_stack.append(action_vector)
        machine_readable_observation = tf.cast(action_stack, dtype=dtype)
        
        return machine_readable_observation, action_dict


    def _stack_list_of_tensors(self, list_of_tensors, dtype=tf.float32):
        stacked_tensor = []
        for tensor in list_of_tensors:
            try:
                # stack each element in tensor
                for el in tensor.numpy():
                    stacked_tensor.append(el)
            except TypeError:
                # tensor only has one element
                stacked_tensor.append(tensor.numpy())
                
        return tf.cast(stacked_tensor, dtype=dtype)








