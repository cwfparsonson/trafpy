from trafpy.generator.src import networks
from trafpy.generator.src import tools
from trafpy.generator.src.demand import Demand

import gym
import tensorflow as tf
import json
import numpy as np
import copy
import pickle
import bz2
import networkx as nx
import queue
import sys
import os
import shutil
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.animation as animation
import cv2
import io
import time
import pandas as pd
from tabulate import tabulate
import pympler
from pympler import tracker
from sqlitedict import SqliteDict

import ray
import psutil
num_cpus = psutil.cpu_count(logical=False)
try:
    ray.init(num_cpus=num_cpus)
except RuntimeError:
    # already initialised ray in script calling dcn sim, no need to init again
    pass


class DCN(gym.Env):

    def __init__(self, 
                 Network, 
                 slots_dict,
                 Scheduler,
                 num_k_paths,
                 env_database_path=None,
                 sim_name='dcn_sim',
                 max_flows=None, 
                 max_time=None,
                 time_multiplexing=True,
                 track_grid_slot_evolution=False,
                 track_queue_length_evolution=False,
                 track_link_utilisation_evolution=True,
                 track_link_concurrent_demands_evolution=True,
                 profile_memory=False,
                 memory_profile_resolution=10,
                 gen_machine_readable_network=False):
        '''
        If time_multiplexing, will assume perfect/ideal time multiplexing where
        can schedule as many different flows per channel so long as sum of flow
        sizes <= maximum channel capacity. If time_multiplexing is False, assume
        no time multiplexing of flows occurs and therefore can only schedule
        one flow per channel.

        slots_dict can either be a slots_dict dictionary, or a str path to a
        pre-defined slots_dict database.

        To reduce memory usage, set tracking grid slot & queue length evolution
        to False

        If env_database_path is not None, will save dicts to database path specified.
        This can significantly reduce simulation RAM memory usage, thereby allowing
        for much larger simulations.

        If gen_machine_readable_network, will generate tensor representation
        of current network state at each step and return it in the obs
        dict. N.B. This process takes a long time (on the order of seconds) and
        will therefore greatly increase simulation time.

        max_time -> time at which to terminate simulation. If None, will run
        simulation untill all flows have both arrived and been completed. If
        'last_demand_arrival_time', will terminate when last demand arrives.

        If profile_memory, will print out summary of memory usage change compared to previous print 
        every memory_profile_resolution % periods of the
        total simulation time as defined by max_time. E.g. if memory_profile_resolution=10,
        will profile memory usage every 10% of max_time.
        '''
        self.sim_name = sim_name 
        print('\nInitialising simulation \'{}\'...'.format(self.sim_name))

        self.profile_memory = profile_memory
        self.memory_profile_resolution = memory_profile_resolution

        if self.profile_memory:
            print('profile_memory set to True. WARNING: Memory profiling can significantly increase simulation time, especially if memory_profile_resolution is low.')
            self.percent_sim_times_profiled = []
            print('Snapshotting memory...')
            start = time.time()
            self.tracker = tracker.SummaryTracker()
            self.tracker.print_diff()
            end = time.time()
            print('Snapshotted memory in {} s'.format(end-start))

        if env_database_path is not None:
            env_database_path += '/env_database'
            if os.path.exists(env_database_path):
                # delete dir
                shutil.rmtree(env_database_path)
            # create dir
            os.mkdir(env_database_path)
        self.env_database_path = env_database_path



        # init slots dict
        self.slots_dict = slots_dict
        if self.env_database_path is not None:
            # create slots dict database
            _slots_dict = self.env_database_path + '/slots_dict.sqlite'
            print('Establishing {} slots_dict tmp database...'.format(self.sim_name))
            start = time.time()
            if type(self.slots_dict) is str:
                print('Slots dict database already created. Copying database over to tmp database dir...')
                shutil.copyfile(self.slots_dict, _slots_dict)
                with SqliteDict(self.slots_dict) as slots_dict:
                    self.slot_size = slots_dict['slot_size']
                    self.job_centric = slots_dict['job_centric']
                    self.num_demands = slots_dict['num_demands']
                    self.num_flows = slots_dict['num_flows']
                    slots_dict.close()
            else:
                print('No slots_dict database path given, only given slots_dict in memory. Creating slots_dict database...')
                with SqliteDict(self.slots_dict) as slots_dict:
                    for key, val in slots_dict.items():
                        if type(key) is not str:
                            slots_dict[json.dumps(key)] = val
                        else:
                            slots_dict[key] = val
                    self.check_if_pairs_valid(slots_dict)
                    self.slot_size = slots_dict['slot_size']
                    self.job_centric = slots_dict['job_centric']
                    self.num_demands = slots_dict['num_demands']
                    self.num_flows = slots_dict['num_flows']
                    slots_dict.commit()
                    slots_dict.close()
            self.slots_dict = _slots_dict # update path to new slots_dict database path
            end = time.time()
            print('Established {} slots_dict tmp database in {} s.'.format(self.sim_name, end-start))
        else:
            # read into memory
            if type(self.slots_dict) is str:
                # slots_dict is database, read into memory
                with SqliteDict(self.slots_dict) as slots_dict:
                    for key, val in slots_dict.items():
                        if type(key) is not str:
                            slots_dict[json.dumps(key)] = val
                        else:
                            slots_dict[key] = val
                    self.check_if_pairs_valid(slots_dict)
                    self.slot_size = slots_dict['slot_size']
                    self.job_centric = slots_dict['job_centric']
                    self.num_demands = slots_dict['num_demands']
                    self.num_flows = slots_dict['num_flows']
                    slots_dict.commit()
                    slots_dict.close()
                # read into memory
                self.slots_dict = slots_dict
            else:
                # slots_dict already read into memory
                self.check_if_pairs_valid(self.slots_dict)
                self.slot_size = self.slots_dict['slot_size']
                self.job_centric = self.slots_dict['job_centric']
                self.num_demands = self.slots_dict['num_demands']
                self.num_flows = slots_dict['num_flows']
            self.check_if_pairs_valid(self.slots_dict)
            self.slot_size = self.slots_dict['slot_size']

        if type(self.slot_size) is not float:
            raise Exception('slot_size must be float (e.g. 1.0), but is {}'.format(self.slot_size))

        # initialise DCN environment characteristics
        self.network = Network
        self.scheduler = Scheduler
        self.num_k_paths = num_k_paths
        self.max_flows = max_flows # max number of flows per queue
        self.max_time = max_time
        if self.max_time == 'last_demand_arrival_time':
            if self.env_database_path is not None:
                with SqliteDict(self.slots_dict) as slots_dict:
                    self.max_time = slots_dict['time_last_demand_arrived']
                slots_dict.close()
            else:
                self.max_time = self.slots_dict['time_last_demand_arrived']

        
        self.time_multiplexing = time_multiplexing
        self.track_grid_slot_evolution = track_grid_slot_evolution
        self.track_queue_length_evolution = track_queue_length_evolution
        self.track_link_utilisation_evolution = track_link_utilisation_evolution
        self.track_link_concurrent_demands_evolution = track_link_concurrent_demands_evolution
        self.gen_machine_readable_network = gen_machine_readable_network

        self.channel_names = self.network.graph['channel_names'] 
        self.num_channels = len(self.channel_names)

        # init representation generator
        if self.gen_machine_readable_network:
            with tf.device('/cpu'):
                self.repgen = RepresentationGenerator(self)

            # gym env reqs
            #         'src': spaces.Box(low=0, high=1, shape=(len(env.repgen.onehot_endpoints[0]),)), # don't need to onehot encode, gym.spaces.Discrete() does automatically
            #         'path': spaces.MultiBinary(nlen(env.repgen.onehot_paths[0])), #TODO: Use this for encoding paths?
            network_representation_space = gym.spaces.Dict({
                index: gym.spaces.Dict({
                        'src': gym.spaces.Discrete(self.repgen.num_endpoints),
                        'dst': gym.spaces.Discrete(self.repgen.num_endpoints),
                        'path': gym.spaces.Discrete(self.repgen.num_paths),
                        'size': gym.spaces.Box(low=-1, high=1e12, shape=()),
                        'packets': gym.spaces.Box(low=-1, high=1e12, shape=()),
                        'time_arrived': gym.spaces.Box(low=-1, high=1e12, shape=()),
                        'selected': gym.spaces.Discrete(2),
                        'null_action': gym.spaces.Discrete(2),
                        'flow_present': gym.spaces.Discrete(2)
                    })
                for index in range(self.repgen.num_actions)})
            self.action_space = gym.spaces.Discrete(self.repgen.num_actions)
            self.observation_space = gym.spaces.Dict({'avail_actions': network_representation_space,
                                                      'machine_readable_network': network_representation_space})

        print('Initialised simulation {}.'.format(self.sim_name))


    def reset(self, return_obs=True):
        '''
        Resets DCN simulation environment
        '''
        print('Resetting simulation \'{}\'...'.format(self.sim_name))

        self.curr_step = 0
        self.curr_time = 0

        self.num_endpoints = int(len(self.network.graph['endpoints']))

        self.net_node_positions = networks.init_network_node_positions(copy.deepcopy(self.network))
        self.animation_images = []

        self.action = {'chosen_flows': []} # init

        if self.job_centric:
            # init dicts & lists required for job centric simulations
            if self.env_database_path is not None:
                # create databases
                self.arrived_job_dicts = self.env_database_path + '/arrived_job_dicts.sqlite'
                with SqliteDict(self.arrived_job_dicts) as arrived_job_dicts:
                    arrived_job_dicts.commit()
                    arrived_job_dicts.close()
                self.completed_job_dicts = self.env_database_path + '/completed_job_dicts.sqlite'
                with SqliteDict(self.completed_job_dicts) as completed_job_dicts:
                    completed_job_dicts.commit()
                    completed_job_dicts.close()
                self.dropped_job_dicts = self.env_database_path + '/dropped_job_dicts.sqlite'
                with SqliteDict(self.dropped_job_dicts) as dropped_job_dicts:
                    dropped_job_dicts.commit()
                    dropped_job_dicts.close()
                self.control_deps = self.env_database_path + '/control_deps.sqlite'
                with SqliteDict(self.control_deps) as control_deps:
                    control_deps.commit()
                    control_deps.close()
            else:
                # use local memory
                # self.arrived_job_dicts = []
                self.arrived_job_dicts = {}
                # self.completed_jobs = []
                self.completed_job_dicts = {}
                # self.dropped_jobs = []
                self.dropped_job_dicts = {}
                # self.control_deps = [] # list of control dependencies
                self.control_deps = {} # list of control dependencies
                # self.control_deps_that_were_flows = []
            self.network.graph['queued_jobs'] = [] # init list of curr queued jobs in network
            self.arrived_jobs = {} # use hash table for quick look ups
            self.running_ops = {}
            self.num_arrived_control_deps = 0
            self.num_completed_control_deps = 0
            self.num_arrived_jobs = 0
            self.num_completed_jobs = 0
            self.num_dropped_jobs = 0
        else:
            # flow centric dicts and lists also needed for job centric -> init below
            pass

        if self.env_database_path is not None:
            # create databases
            self.arrived_flow_dicts = self.env_database_path + '/arrived_flow_dicts.sqlite'
            with SqliteDict(self.arrived_flow_dicts) as arrived_flow_dicts:
                arrived_flow_dicts.commit()
                arrived_flow_dicts.close()
            self.completed_flow_dicts = self.env_database_path + '/completed_flow_dicts.sqlite'
            with SqliteDict(self.completed_flow_dicts) as completed_flow_dicts:
                completed_flow_dicts.commit()
                completed_flow_dicts.close()
            self.dropped_flow_dicts = self.env_database_path + '/dropped_flow_dicts.sqlite'
            with SqliteDict(self.dropped_flow_dicts) as dropped_flow_dicts:
                dropped_flow_dicts.commit()
                dropped_flow_dicts.close()

        else:
            # use local memory
            self.arrived_flow_dicts = {}
            self.completed_flow_dicts = {}
            self.dropped_flow_dicts = {}
        self.arrived_flows = {}
        self.connected_flows = []
        self.num_arrived_flows = 0
        self.num_completed_flows = 0
        self.num_dropped_flows = 0


        self.network = self.init_virtual_queues(self.network)
        if self.track_queue_length_evolution:
            self.queue_evolution_dict = self.init_queue_evolution(self.network)
        if self.track_grid_slot_evolution:
            self.grid_slot_dict = self.init_grid_slot_evolution(self.network)
        if self.track_link_utilisation_evolution:
            if self.env_database_path is not None:
                # create link util dict database
                self.link_utilisation_dict = self.env_database_path + '/link_utilisation_dict.sqlite'
                with SqliteDict(self.link_utilisation_dict) as link_utilisation_dict:
                    for key, val in self.init_link_utilisation_evolution(self.network).items():
                        link_utilisation_dict[key] = val
                    link_utilisation_dict.commit()
                    link_utilisation_dict.close()
            else:
                # read into memory
                self.link_utilisation_dict = self.init_link_utilisation_evolution(self.network)
        if self.track_link_concurrent_demands_evolution:
            if self.env_database_path is not None:
                # create link concurrent demands dict database
                self.link_concurrent_demands_dict = self.env_database_path + '/link_concurrent_demands_dict.sqlite'
                with SqliteDict(self.link_concurrent_demands_dict) as link_concurrent_demands_dict:
                    for key, val in self.init_link_concurrent_demands_dict(self.network).items():
                        link_concurrent_demands_dict[key] = val
                    link_concurrent_demands_dict.commit()
                    link_concurrent_demands_dict.close()
            else:
                # read into memory
                self.link_concurrent_demands_dict = self.init_link_concurrent_demands_dict(self.network)

        print('Reset simulation {}.'.format(self.sim_name))
        



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
        ran.
        '''
        key = list(slots_dict['slot_keys'])[0]
        slot = slots_dict[key]
        for event in slot['new_event_dicts']:
            if self.job_centric:
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
                # queued_flow_bytes = sum(flow_dict['packets'])
                queued_flow_bytes = flow_dict['packets']*flow_dict['packet_size']
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

    def get_channel_bandwidth(self, edge, channel):
        '''Gets current channel bandwidth left on a given edge in the network.'''
        try:
            # return self.network[edge[0]][edge[1]]['channels'][channel]
            return self.network[edge[0]][edge[1]]['{}_to_{}_port'.format(edge[0], edge[1])]['channels'][channel]
        except KeyError:
            # return self.network[edge[1]][edge[0]]['channels'][channel]
            return self.network[edge[1]][edge[0]]['{}_to_{}_port'.format(edge[0], edge[1])]['channels'][channel]

    def init_grid_slot_evolution(self, Graph):
        grid_slot_dict = {ep:
                            {channel:
                                {'times': [0],
                                 'demands': [None],
                                 'demands_info': [None]}
                             for channel in self.channel_names} 
                          for ep in Graph.graph['endpoints']}

        return grid_slot_dict
    
    def init_link_utilisation_evolution(self, net):
        link_utilisation_dict = {}
        for link in net.edges:
            # src-dst
            link_utilisation_dict[json.dumps([link[0], link[1]])] = {'time_slots': [self.curr_step],
                                                                     'util': [0]}
            # dst-src
            link_utilisation_dict[json.dumps([link[1], link[0]])] = {'time_slots': [self.curr_step],
                                                                     'util': [0]}

        return link_utilisation_dict

    def init_link_concurrent_demands_dict(self, net):
        link_concurrent_demands_dict = {}
        for link in net.edges:
            # src-dst
            link_concurrent_demands_dict[json.dumps([link[0], link[1]])] = {'time_slots': [self.curr_step],
                                                                            'concurrent_demands': [0]}
            # dst-src
            link_concurrent_demands_dict[json.dumps([link[1], link[0]])] = {'time_slots': [self.curr_step],
                                                                            'concurrent_demands': [0]}

        return link_concurrent_demands_dict




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

    def check_chosen_flows_valid(self, chosen_flows):
        self.time_check_valid_start = time.time()

        # init channel link occupation dict
        edge_channel_occupation = {json.dumps(edge): 
                                    {channel: 'unoccupied' for channel in self.channel_names}
                                   for edge in self.network.edges}

        # update any channel links which are now occupied 
        for flow in chosen_flows:
            path, channel = flow['path'], flow['channel']
            edges = self.get_path_edges(path)
            for edge in edges:
                try:
                    if edge_channel_occupation[json.dumps(edge)][channel] != 'unoccupied':
                        raise Exception('Scheduler chose flow {}, however at least one of the edge channels in this chosen path-channel is already occupied by flow {}. Resolve contentions before passing chosen flows to DCN simulation environment.'.format(flow, edge_channel_occupation[json.dumps(edge)][channel]))
                    edge_channel_occupation[json.dumps(edge)][channel] = flow
                except KeyError:
                    if edge_channel_occupation[json.dumps(edge[::-1])][channel] != 'unoccupied':
                        raise Exception('Scheduler chose flow {}, however at least one of the edge channels in this chosen path-channel is already occupied by flow {}. Resolve contentions before passing chosen flows to DCN simulation environment.'.format(flow, edge_channel_occupation[json.dumps(edge[::-1])][channel]))
                    edge_channel_occupation[json.dumps(edge[::-1])][channel] = flow

        self.time_check_valid_end = time.time()

    def update_link_utilisation_evolution(self):
        if self.env_database_path is not None:
            link_utilisation_dict = SqliteDict(self.link_utilisation_dict)
        else:
            link_utilisation_dict = self.link_utilisation_dict

        for link in link_utilisation_dict.keys():
            link = json.loads(link)
            # src-dst
            available_link_bw = 0
            max_link_bw = 0
            for channel in self.channel_names:
                available_link_bw += self.get_channel_bandwidth(link, channel)
                max_link_bw += self.network[link[0]][link[1]]['{}_to_{}_port'.format(link[0], link[1])]['max_channel_capacity']
            link_util = 1-(available_link_bw / max_link_bw)
            # get link dict
            link_dict = link_utilisation_dict[json.dumps(link)]
            # update dict
            link_dict['time_slots'].append(self.curr_step)
            link_dict['util'].append(link_util)
            link_utilisation_dict[json.dumps(link)] = link_dict

            # dst-src
            link = link[::-1]
            available_link_bw = 0
            max_link_bw = 0
            for channel in self.channel_names:
                available_link_bw += self.get_channel_bandwidth(link, channel)
                max_link_bw += self.network[link[0]][link[1]]['{}_to_{}_port'.format(link[0], link[1])]['max_channel_capacity']
            link_util = 1-(available_link_bw / max_link_bw)
            # get link dict
            link_dict = link_utilisation_dict[json.dumps(link)]
            # update dict
            link_dict['time_slots'].append(self.curr_step)
            link_dict['util'].append(link_util)
            link_utilisation_dict[json.dumps(link)] = link_dict

        if self.env_database_path is not None:
            link_utilisation_dict.commit()
            link_utilisation_dict.close()
        else:
            self.link_utilisation_dict = link_utilisation_dict


    def update_link_concurrent_demands_evolution(self, link, num_concurrent_demands_to_add=1):
        '''Adds num_concurrent_demands_to_add to current number of concurrent demands on a given link.'''
        if self.env_database_path is not None:
            link_concurrent_demands_dict = SqliteDict(self.link_concurrent_demands_dict)
        else:
            link_concurrent_demands_dict = self.link_concurrent_demands_dict

        link_dict = link_concurrent_demands_dict[json.dumps(link)]
        if link_concurrent_demands_dict[json.dumps(link)]['time_slots'][-1] != self.curr_step:
            # not yet evolved concurrent demands evolution for this time slot
            link_dict['time_slots'].append(self.curr_step)
            # init num concurrent demands on link for this time slot
            link_dict['concurrent_demands'].append(0)
        
        # update num concurrent demands tracker for this time slot
        link_dict['concurrent_demands'][-1] += num_concurrent_demands_to_add
        link_concurrent_demands_dict[json.dumps(link)] = link_dict

        if self.env_database_path is not None:
            link_concurrent_demands_dict.commit()
            link_concurrent_demands_dict.close()
        else:
            self.link_concurrent_demands_dict = link_concurrent_demands_dict





    def update_grid_slot_evolution(self, chosen_flows):
        time = self.curr_time
                    
        # init ep channel link occupation dict
        ep_link_occupation = {ep: 
                                {channel: 'unoccupied' for channel in self.channel_names}
                             for ep in self.network.graph['endpoints']}

        # update grid slot evolution with any ep link channels which are now occupied 
        if 'unique_id' in chosen_flows[0]:
            identifier = 'unique_id'
        else:
            identifier = 'flow_id'

        for flow in chosen_flows:
            sn, dn, channel = flow['src'], flow['dst'], flow['channel']
            if sn == dn:
                # does not become flow therefore does not occupy an ep channel link
                pass
            else:
                # src ep link
                self.grid_slot_dict[sn][channel]['demands'].append(flow[identifier])
                self.grid_slot_dict[sn][channel]['times'].append(time)
                self.grid_slot_dict[sn][channel]['demands_info'].append(flow) # useful for debugging
                ep_link_occupation[sn][channel] = 'occupied'
                # dst ep link
                self.grid_slot_dict[dn][channel]['demands'].append(flow[identifier])
                self.grid_slot_dict[dn][channel]['times'].append(time)
                self.grid_slot_dict[dn][channel]['demands_info'].append(flow) # useful for debugging
                ep_link_occupation[dn][channel] = 'occupied'

        # update grid slot evolution of any ep link channels which are unoccupied
        for ep in ep_link_occupation.keys():
            for channel in ep_link_occupation[ep].keys():
                if ep_link_occupation[ep][channel] == 'occupied':
                    # ep link channel occupied, grid slot has already been updated
                    pass
                else:
                    # ep link channel unoccupied, grid slot not yet updated
                    self.grid_slot_dict[ep][channel]['demands'].append(None)
                    self.grid_slot_dict[ep][channel]['times'].append(time)
                    self.grid_slot_dict[ep][channel]['demands_info'].append(None) # useful for debugging








    
    def init_virtual_queues(self, Graph):
        # queues_per_ep = self.num_endpoints-1
      
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
        if 'unique_id' in flow_dict:
            identifier = 'unique_id'
        else:
            identifier = 'flow_id'

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
            if self.env_database_path is not None:
                # if self.job_centric:
                    # dropped_id = flow_dict['job_id']+'_'+flow_dict['flow_id']
                # else:
                    # dropped_id = flow_dict['flow_id']
                with SqliteDict(self.dropped_flow_dicts) as dropped_flow_dicts:
                    # dropped_flow_dicts[flow_dict['flow_id']] = flow_dict
                    # dropped_flow_dicts[dropped_id] = flow_dict
                    dropped_flow_dicts[flow_dict[identifier]] = flow_dict
                    dropped_flow_dicts.commit()
                    dropped_flow_dicts.close()
            else:
                # self.dropped_flow_dicts[flow_dict['flow_id']] = flow_dict
                # self.dropped_flow_dicts[dropped_id] = flow_dict
                self.dropped_flow_dicts[flow_dict[identifier]] = flow_dict
            self.num_dropped_flows += 1
            if self.job_centric:
                for job_dict in self.network.graph['queued_jobs']:
                    if job_dict['job_id'] == flow_dict['job_id']:
                        # drop job
                        # self.dropped_jobs.append(job_dict)
                        if self.env_database_path is not None:
                            with SqliteDict(self.dropped_job_dicts) as dropped_job_dicts:
                                dropped_job_dicts[job_dict['job_id']] = job_dict
                                dropped_job_dicts.commit()
                                dropped_job_dicts.close()
                        else:
                            self.dropped_job_dicts[job_dict['job_id']] = job_dict
                        self.num_dropped_jobs += 1
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
        self.num_arrived_jobs += 1
        # self.arrived_job_dicts.append(job_dict)
        if self.env_database_path is not None:
            with SqliteDict(self.arrived_job_dicts) as arrived_job_dicts:
                arrived_job_dicts[job_dict['job_id']] = job_dict
                arrived_job_dicts.commit()
                arrived_job_dicts.close()
        else:
            self.arrived_job_dicts[job_dict['job_id']] = job_dict
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
                    if int(flow_dict['parent_op_run_time']) == 0 and len(flow_dict['completed_parent_deps']) == len(flow_dict['parent_deps']):
                        # control dependency satisfied immediately
                        flow_dict['time_arrived'] = self.curr_time
                        flow_dict['time_completed'] = self.curr_time
                        flow_dict['can_schedule'] = 1
                        flows_to_complete.append(flow_dict)
                    # add control dependency to arrived control dependencies
                    if self.env_database_path is not None:
                        with SqliteDict(self.control_deps) as control_deps:
                            control_deps[flow_dict['unique_id']] = flow_dict
                            control_deps.commit()
                            control_deps.close()
                    else:
                        self.control_deps[flow_dict['unique_id']] = flow_dict
                    self.num_arrived_control_deps += 1
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

        if self.env_database_path is not None:
            control_deps = SqliteDict(self.control_deps)
        else:
            control_deps = self.control_deps

        # go through queued control dependencies
        _control_deps = {} # tmp for updated control deps so don't crash database by updating while looping through
        # must update control_deps dict outside of loop 
        # or will crash database, therefore register any completed flows
        # (which requires accessing control_deps to check flow/dep completed)
        # outside of loop
        deps_to_complete = []
        for key in control_deps.keys():
            dep = control_deps[key]
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
                    # dep['time_completed'] = self.curr_time
                    if dep['time_completed'] is None:
                        dep['time_completed'] = self.curr_time + self.slot_size
                        dep['can_schedule'] = 1
                        deps_to_complete.append(dep)
                        # self.register_completed_flow(dep)

                        # store update
                        _control_deps[dep['unique_id']] = dep
                    else:
                        # already registed completed
                        pass

                else:
                    # parent op not yet finished
                    pass
            else:
                # parent op not yet started
                pass

        # update with stored updates
        for key, val in _control_deps.items():        
            control_deps[key] = val

        if self.env_database_path is not None:
            control_deps.commit()
            control_deps.close()
        else:
            self.control_deps = control_deps

        # complete any deps to complete
        for dep in deps_to_complete:
            self.register_completed_flow(dep)

        observation['network'] = self.network

        return observation

                        
    def add_flows_to_queues(self, observation):
        '''
        Takes observation of current time slot and updates virtual queues in
        network
        '''
        self.time_queue_flows_start = time.time()

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
                    if self.job_centric:
                        self.add_job_to_queue(event_dict, print_times=False)
                    else:
                        self.add_flow_to_queue(event_dict)
        
        observation['network'] = self.network # update osbervation's network

        self.time_queue_flows_end = time.time()
        
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
        if 'unique_id' in flow_dict:
            identifier = 'unique_id'
        else:
            identifier = 'flow_id'
      
        # record time at which flow was completed
        start = time.time()
        # flow_dict['time_completed'] = copy.copy(self.curr_time)
        # if self.job_centric:
            # completion_id = flow_dict['job_id']+'_'+flow_dict['flow_id']
        # else:
            # completion_id = flow_dict['flow_id']
        flow_dict['time_completed'] = copy.copy(self.curr_time) + self.slot_size
        if flow_dict['size'] != 0 and flow_dict['src'] != flow_dict['dst']:
            # flow was an actual flow
            if self.env_database_path is not None:
                with SqliteDict(self.completed_flow_dicts) as completed_flow_dicts:
                    # completed_flow_dicts[flow_dict['flow_id']] = flow_dict
                    # completed_flow_dicts[completion_id] = flow_dict
                    completed_flow_dicts[flow_dict[identifier]] = flow_dict
                    completed_flow_dicts.commit()
                    completed_flow_dicts.close()
            else:
                # self.completed_flow_dicts[flow_dict['flow_id']] = flow_dict
                # self.completed_flow_dicts[completion_id] = flow_dict
                self.completed_flow_dicts[flow_dict[identifier]] = flow_dict
            self.num_completed_flows += 1
        else:
            # 'flow' never actually became a flow (src == dst or control dependency)
            self.num_completed_control_deps += 1
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
        if self.job_centric:
            # make any necessary job completion & job dependency changes
            self.update_completed_flow_job(f)
        end = time.time()
        if print_times:
            print('Time to record any job completions and job dependency changes: {}'.format(end-start))


    def register_arrived_flow(self, flow_dict):
        if 'unique_id' in flow_dict:
            identifier = 'unique_id'
        else:
            identifier = 'flow_id'

        # register
        if flow_dict['can_schedule'] == 1:
            # flow is ready to be scheduled therefore can count as arrived
            # if self.job_centric:
                # arrival_id = flow_dict['job_id']+'_'+flow_dict['flow_id']
            # else:
                # arrival_id = flow_dict['flow_id']
            if flow_dict[identifier] in self.arrived_flows:
                # flow already counted as arrived
                pass
            else:
                # flow not yet counted as arrived
                if flow_dict['src'] != flow_dict['dst'] and flow_dict['size'] != 0:
                    if flow_dict['time_arrived'] is None:
                        # record time flow arrived
                        flow_dict['time_arrived'] = self.curr_time
                    else:
                        # already recorded time of arrival
                        pass
                    # self.arrived_flows[arrival_id] = 'present'
                    self.arrived_flows[flow_dict[identifier]] = 'present'
                    if self.env_database_path is not None:
                        with SqliteDict(self.arrived_flow_dicts) as arrived_flow_dicts:
                            arrived_flow_dicts[flow_dict[identifier]] = flow_dict
                            arrived_flow_dicts.commit()
                            arrived_flow_dicts.close()
                    else:
                        # self.arrived_flow_dicts[arrival_id] = flow_dict
                        self.arrived_flow_dicts[flow_dict[identifier]] = flow_dict
                    self.num_arrived_flows += 1
                else:
                    # 'flow' never actually becomes flow (is ctrl dependency or src==dst)
                    pass
        else:
            # can't yet schedule therefore don't count as arrived
            pass

        
    def get_max_flow_info_transferred_per_slot(self, flow_dict):
        '''
        Returns maximum possible flow information & number of packets transferred
        per timeslot given the flow's path (i.e. in point-to-point circuit switched
        network, max info transferred per slot is the bandwidth of the lowest bw
        link in the path * the slot size)
        '''
        packet_size = flow_dict['packet_size']
        path_links = self.get_path_edges(flow_dict['path'])
        channel = flow_dict['channel']
        link_bws = []
        for link in path_links:
            # link_bws.append(self.network[link[0]][link[1]]['channels'][channel])
            link_bws.append(self.network[link[0]][link[1]]['{}_to_{}_port'.format(link[0], link[1])]['channels'][channel])
        capacity = min(link_bws) # channel capacity == info transferred per unit time
        info_per_slot = capacity  * self.slot_size # info transferred per slot == info transferred per unit time * number of time units (i.e. slot size)
        packets_per_slot = int(info_per_slot / packet_size) # round down 

        return info_per_slot, packets_per_slot


    def update_flow_packets(self, flow_dict):
        '''
        Takes flow dict that has been schedueled to be activated for curr
        time slot and removes corresponding number of packets flow in queue
        '''
        # info_per_slot, packets_per_slot = self.get_max_flow_info_transferred_per_slot(flow_dict)
        # if flow_dict['packets_this_slot'] > packets_per_slot:
            # raise Exception('Trying to transfer {} packets this slot, but flow {} can only have up to {} packets transferred per slot.'.format(flow_dict['packets_this_slot'], flow_dict, packets_per_slot))

        
        sn = flow_dict['src']
        dn = flow_dict['dst']
        queued_flows = self.network.nodes[sn][dn]['queued_flows']
        idx = self.find_flow_idx(flow_dict, queued_flows)
        queued_flows[idx]['packets'] -= flow_dict['packets_this_slot']
        queued_flows[idx]['packets_this_slot'] = flow_dict['packets_this_slot']
        if queued_flows[idx]['packets'] < 0:
            queued_flows[idx]['packets'] = 0
        
        updated_flow = copy.copy(queued_flows[idx])
        if updated_flow['packets'] == 0:
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
        self.time_next_obs_start = time.time()
        try:
            if self.env_database_path is not None:
                # read from database
                with SqliteDict(self.slots_dict) as slots_dict:
                    observation = {'slot_dict': slots_dict[json.dumps(self.curr_step)],
                                   'network': copy.deepcopy(self.network)}
                    self.update_curr_time(observation['slot_dict'])
                    slots_dict.close()
            else:
                # stored in memory
                observation = {'slot_dict': self.slots_dict[self.curr_step],
                               'network': copy.deepcopy(self.network)}
                self.update_curr_time(observation['slot_dict'])
            # add any new events (flows or jobs) to queues
            observation = self.add_flows_to_queues(observation)
            # save step used as most recent valid curr_step for slots_dict indexing
            self.most_recent_valid_curr_step = self.curr_step
        except KeyError:
            # curr step either exceeded slots dict indices (no new flows/jobs arriving) or this step was not included in slots_dict since no demands arrived
            # index slot_dict with most recent valid curr step
            if self.env_database_path is not None:
                # read from database
                with SqliteDict(self.slots_dict) as slots_dict:
                    observation = {'slot_dict': slots_dict[json.dumps(self.most_recent_valid_curr_step)],
                                   'network': copy.deepcopy(self.network)}
                    self.update_curr_time(observation['slot_dict'])
                    slots_dict.close()
            else:
                # stored in memory
                observation = {'slot_dict': self.slots_dict[self.most_recent_valid_curr_step],
                               'network': copy.deepcopy(self.network)}
                self.update_curr_time(observation['slot_dict'])
        
        # update any dependencies of running ops
        if self.job_centric:
            observation = self.update_running_op_dependencies(observation)

        if self.gen_machine_readable_network:
            with tf.device('/cpu'):
                # create machine readable version of current network state
                _, observation['machine_readable_network'] = self.repgen.gen_machine_readable_network_observation(observation['network'], dtype=tf.float16)

                # update available actions
                observation['avail_actions'] = observation['machine_readable_network']

        self.time_next_obs_end = time.time()


            
        return observation

                



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
                    if self.job_centric:
                        f_id = str(flow['job_id']+'_'+flow['flow_id'])
                    else:
                        f_id = str(flow['flow_id'])
                    network.add_node(f_id)
                    network.add_edge(f_id, flow['src'], type='queue_link')
                    flows.append(f_id)
                    pos[f_id] = (list(pos[flow['src']])[0]+appended_node_x_spacing, list(pos[flow['src']])[1]+y_offset)
                    y_offset-=appended_node_y_spacing

        if self.job_centric:
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
            if 'channels' in network[edge[0]][edge[1]]['{}_to_{}_port'.format(edge[0], edge[1])].keys():
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

        if draw_ops and self.job_centric:
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
        # job_dict['time_completed'] = copy.copy(self.curr_time)
        job_dict['time_completed'] = copy.copy(self.curr_time) + self.slot_size
        self.num_completed_jobs += 1
        # self.completed_jobs.append(job_dict)
        if self.env_database_path is not None:
            with SqliteDict(self.completed_job_dicts) as completed_job_dicts:
                completed_job_dicts[job_dict['job_id']] = job_dict
                completed_job_dicts.commit()
                completed_job_dicts.close()
        else:
            self.completed_job_dicts[job_dict['job_id']] = job_dict
        # jct = job_dict['time_completed']-job_dict['time_arrived']
        
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
        if self.env_database_path is not None:
            control_deps = SqliteDict(self.control_deps)
        else:
            control_deps = self.control_deps

        for dep in control_deps.values():
            if dep['job_id'] == job_id and dep['time_completed'] is None:
                # job still has at least one uncompleted control dependency left
                job_ctrl_deps_completed = False
                break

        if self.env_database_path is not None:
            control_deps.close()

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

        if self.env_database_path is not None:
            control_deps = SqliteDict(self.control_deps)
        else:
            control_deps = self.control_deps
        _control_deps = {} # tmp dict so don't need to edit dict during loop (crashes database)
        
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
                for key in control_deps.keys():
                    control_dep = control_deps[key]
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
                            # store update
                            _control_deps[control_dep['unique_id']] = control_dep
                        else:
                            # dep is not a child dep of completed flow
                            pass
                    else:
                        # dep not part of same job as completed flow
                        pass

        # update with stored updates
        for key, val in _control_deps.items():
            control_deps[key] = val

        if self.env_database_path is not None:
            control_deps.commit()
            control_deps.close()
        else:
            self.control_deps = control_deps
                    

    def update_flow_attrs(self, chosen_flows):
        self.time_update_flow_attrs_start = time.time()

        for flow in chosen_flows:
            sn = flow['src']
            dn = flow['dst']
            queued_flows = self.network.nodes[sn][dn]['queued_flows']
            idx = self.find_flow_idx(flow, queued_flows)
            dated_flow = self.network.nodes[sn][dn]['queued_flows'][idx]
            dated_flow['packets_this_slot'] = flow['packets_this_slot']
            if dated_flow['packets'] is None:
                # udpate flow packets and k shortest paths
                dated_flow['packets'] = flow['packets']
                dated_flow['packet_size'] = flow['packet_size']
                dated_flow['k_shortest_paths'] = flow['k_shortest_paths']
            else:
                # agent updates already applied
                pass

        self.time_update_flow_attrs_end = time.time()

    def reset_channel_capacities_of_edges(self, edges):
        '''Takes edges and resets their available capacities back to their maximum capacities.'''
        for edge in edges:
            # for channel in self.network[edge[0]][edge[1]]['channels']:
            for channel in self.network[edge[0]][edge[1]]['{}_to_{}_port'.format(edge[0], edge[1])]['channels']:
                # reset channel capacity of both ports
                self.network[edge[0]][edge[1]]['{}_to_{}_port'.format(edge[0], edge[1])]['channels'][channel] = self.network[edge[0]][edge[1]]['{}_to_{}_port'.format(edge[0], edge[1])]['max_channel_capacity']
                self.network[edge[1]][edge[0]]['{}_to_{}_port'.format(edge[1], edge[0])]['channels'][channel] = self.network[edge[0]][edge[1]]['{}_to_{}_port'.format(edge[0], edge[1])]['max_channel_capacity']
        # update global graph property
        self.network.graph['curr_nw_capacity_used'] = 0


    def take_action(self, action):
        self.time_take_action_start = time.time()

        # reset channel capacities of all links (start with no flows scheduled therefore all channel capacity is available before action is taken)
        self.reset_channel_capacities_of_edges(self.network.edges)

        # unpack chosen action
        chosen_flows = action['chosen_flows']

        if not self.time_multiplexing:
            # no time multiplexing -> can only schedule one flow per channel per time slot. Check chosen flows valid
            self.check_chosen_flows_valid(chosen_flows)
        else:
            # assume perfect time multiplexing -> can schedule as many flow packets as possible so long as sum(scheduled_flows_packets) <= max_channel_capacity. If chosen flows are not valid, an exception will be raised later when trying to setup the flows.
            pass
        
        # update any flow attrs in dcn network that have been updated by agent
        self.update_flow_attrs(chosen_flows)

        # establish chosen flows
        self.time_establish_flows_start = time.time()
        # if len(chosen_flows) == 0:
            # # no flows chosen
            # pass
        # else:
            # ray.get([self.set_up_connection.remote(self, flow) for flow in chosen_flows])
        for chosen_flow in chosen_flows:
            # chosen flow not already established, establish connection
            self.set_up_connection(chosen_flow)
        self.time_establish_flows_end = time.time()

        # take down replaced flows
        self.time_takedown_flows_start = time.time()
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
        self.time_takedown_flows_end = time.time()



        # all chosen flows established and any removed flows taken down
        # update connected_flows

        self.connected_flows = chosen_flows.copy()

        if self.track_queue_length_evolution:
            self.update_queue_evolution()
        if self.track_grid_slot_evolution:
            self.update_grid_slot_evolution(chosen_flows)
        if self.track_link_utilisation_evolution:
            self.update_link_utilisation_evolution()
        # N.B. update_link_concurrent_demands_evolution() done inside set_up_connection() for efficiency

        self.time_take_action_end = time.time()

    def display_step_processing_time(self, num_decimals=8):
        # step
        step_time = self.time_step_end - self.time_step_start

        # take action
        take_action_time = self.time_take_action_end - self.time_take_action_start

        # # check flow validity
        # check_valid_time = self.time_check_valid_end - self.time_check_valid_start

        # update flow attrs
        update_flow_attrs_time = self.time_update_flow_attrs_end - self.time_update_flow_attrs_start

        # establish chosen flows
        establish_flows_time = self.time_establish_flows_end - self.time_establish_flows_start

        # take down replaced flows
        takedown_flows_time = self.time_takedown_flows_end - self.time_takedown_flows_start

        # reward

        # check done

        # next obs
        next_obs_time = self.time_next_obs_end - self.time_next_obs_start

        # queue flows
        if self.most_recent_valid_curr_step == self.curr_step:
            # attempted to add flows to queue
            queue_flows_time = self.time_queue_flows_end - self.time_queue_flows_start
            queued_flows = True
        else:
            # no attempt to add flows to queue
            queued_flows = False

        # gen machine readable
        if self.gen_machine_readable_network:
            try:
                gen_machine_readable_time = self.repgen.time_gen_machine_readable_end - self.repgen.time_gen_machine_readable_start
                machine_readable = True
            except AttributeError:
                # haven't generated a machine readable representation
                machine_readable = False

        # create table
        summary_dict = {
                'Step': [round(step_time, num_decimals)],
                'Take Action': [round(take_action_time, num_decimals)],
                # 'Check Valid': [round(check_valid_time, num_decimals)],
                'Update Attrs': [round(update_flow_attrs_time, num_decimals)],
                'Establish': [round(establish_flows_time, num_decimals)],
                'Takedown': [round(takedown_flows_time, num_decimals)],
                'Next Obs': [round(next_obs_time, num_decimals)]
                }
        if queued_flows:
            summary_dict['Q Flows'] = [round(queue_flows_time, num_decimals)]
        if self.gen_machine_readable_network:
            if machine_readable:
                summary_dict['Gen Mach'] = [round(gen_machine_readable_time, num_decimals)]

        df = pd.DataFrame(summary_dict)
        print('')
        print(tabulate(df, showindex=False, headers='keys', tablefmt='psql'))




    def display_env_memory_usage(self, obs):
        # slots_dict
        slots_dict_size = sys.getsizeof(json.dumps(self.slots_dict))

        # network
        network_size = sys.getsizeof(pickle.dumps(self.network)) 

        # grid slot evolution
        if self.track_grid_slot_evolution:
            grid_slot_size = sys.getsizeof(json.dumps(self.grid_slot_dict))

        if self.track_queue_length_evolution:
            queue_length_size = sys.getsizeof(json.dumps(self.queue_evolution_dict))

        # # machine readable representation
        # machine_readable_network_size = sys.getsizeof(json.dumps(obs['machine_readable_network']))

        # # observation
        # obs_size = sys.getsizeof(json.dumps(obs))

        # create table
        summary_dict = {
                'Time': [self.curr_time],
                'Slots Dict (B)': [slots_dict_size],
                'Network (B)': [network_size]}
        if self.track_grid_slot_evolution:
            summary_dict['Grid Slot (B)'] = [grid_slot_size]
        if self.track_queue_length_evolution:
            summary_dict['Queue Evol (B)'] = [queue_length_size]
        df = pd.DataFrame(summary_dict)
        print('')
        print(tabulate(df, showindex=False, headers='keys', tablefmt='psql'))



    def step(self, action, print_memory_usage=False, print_processing_time=False):
        '''
        Performs an action in the DCN simulation environment, moving simulator
        to next step
        '''
        # # DEBUG:
        # queues = self.get_current_queue_states() 
        # print('\nQueues being given to scheduler:')
        # i = 0
        # for q in queues:
           # print('queue {}:\n{}'.format(i, q))
           # i+=1
        # # print('Chosen action received by environment:\n{}'.format(action))
        # # print('Queued jobs:\n{}'.format(self.network.graph['queued_jobs']))
        # print('Incomplete control deps:')
        # control_deps = SqliteDict(self.control_deps)
        # for c in control_deps.values():
           # if c['time_completed'] is None or c['time_parent_op_started'] is None:
               # print(c)
           # else:
               # pass
        # control_deps.close()
        # if self.job_centric:
            # print('Time: {} Step: {} | Sim demands: {} | Flows arrived/completed/dropped: {}/{}/{} | Jobs arrived/completed/dropped: {}/{}/{} | Ctrl deps arrived/completed: {}/{}'.format(self.curr_time, self.curr_step, self.num_demands, self.num_arrived_flows, self.num_completed_flows, self.num_dropped_flows, self.num_arrived_jobs, self.num_completed_jobs, self.num_dropped_jobs, self.num_arrived_control_deps, self.num_completed_control_deps)) 
        # else:
            # print('Time: {} Step: {} | Sim demands: {} | Flows arrived/completed/dropped: {}/{}/{}'.format(self.curr_time, self.curr_step, self.num_demands, self.num_arrived_flows, self.num_completed_flows, self.num_dropped_flows)) 





        self.time_step_start = time.time()

        self.action = action # save action

        self.take_action(action)

        self.curr_step += 1
        reward = self.calc_reward()
        done = self.check_if_done()
        info = None
        obs = self.next_observation()

        self.time_step_end = time.time()


        if print_memory_usage:
            self.display_env_memory_usage(obs)
        if print_processing_time:
            self.display_step_processing_time()

        if self.profile_memory:
            percent_sim_time_completed = round(100*(self.curr_time/self.max_time), 0)
            if percent_sim_time_completed % self.memory_profile_resolution == 0:
                if percent_sim_time_completed not in self.percent_sim_times_profiled:
                    print('Snapshotting memory...')
                    start = time.time()
                    self.tracker.print_diff()
                    end = time.time()
                    print('Snapshotted in {} s'.format(end-start))
                    self.percent_sim_times_profiled.append(percent_sim_time_completed)


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
        if self.num_arrived_flows == 0:
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
            if self.job_centric:
                if (self.num_arrived_jobs != self.num_completed_jobs and self.num_arrived_jobs != 0 and self.num_demands != self.num_dropped_jobs+self.num_completed_jobs) or self.num_arrived_jobs != self.num_demands:
                    return False
                else:
                    self.check_if_any_flows_arrived()
                    return True
            else:
                if (self.num_arrived_flows != self.num_completed_flows and self.num_arrived_flows != 0 and self.num_demands != self.num_dropped_flows+self.num_completed_flows) or self.num_arrived_flows != self.num_demands:
                    return False
                else:
                    self.check_if_any_flows_arrived()
                    return True

        else:
            if self.job_centric:
                if self.curr_time >= self.max_time:
                    self.check_if_any_flows_arrived()
                    return True
                elif (self.num_arrived_jobs != self.num_completed_jobs and self.num_arrived_jobs != 0 and self.num_demands != self.num_dropped_jobs+self.num_completed_jobs) or self.num_arrived_jobs != self.num_demands:
                    return False
                else:
                    self.check_if_any_flows_arrived()
                    return True
            else:
                if self.curr_time >= self.max_time:
                    self.check_if_any_flows_arrived()
                    return True
                elif (self.num_arrived_flows != self.num_completed_flows and self.num_arrived_flows != 0 and self.num_demands != self.num_dropped_flows+self.num_completed_flows) or self.num_arrived_flows != self.num_demands:
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
            # capacity = graph[node_pair[0]][node_pair[1]]['channels'][channel]
            capacity = graph[node_pair[0]][node_pair[1]]['{}_to_{}_port'.format(node_pair[0], node_pair[1])]['channels'][channel]
            if round(capacity,0) != round(graph[node_pair[0]][node_pair[1]]['{}_to_{}_port'.format(node_pair[0], node_pair[1])]['max_channel_capacity'],0):
                channel_used = True
                break
            else:
                continue

        return channel_used
        
    # @ray.remote
    def set_up_connection(self, flow, num_decimals=6):
        '''
        Sets up connection between src-dst node pair by removing capacity from
        all edges in path connecting them. Also updates graph's global curr 
        network capacity used property
        
        Args:
        - flow (dict): flow dict containing flow info to set up
        '''
        if flow['can_schedule'] == 0:
            raise Exception('Tried to set up flow {}, but this flow cannot yet be scheduled (can_schedule == 0)! Scheduler should not be giving invalid chosen flow sets to the environment.'.format(flow)) 

        path = flow['path']
        channel = flow['channel']
        flow_size = flow['size']
        packet_size = flow['packet_size']
        packets = flow['packets']
        packets_this_slot = flow['packets_this_slot']

        info_to_transfer_this_slot = packets_this_slot * packet_size
        capacity_used_this_slot = round(info_to_transfer_this_slot / self.slot_size, num_decimals) # info units of this flow transferred this time slot == capacity used on each channel in flow's path this time slot


        edges = self.get_path_edges(path)
        num_edges = len(edges)
        for edge in range(num_edges):
            node_pair = edges[edge]

            # check that establishing this flow is valid given edge capacity constraints
            if self.network[node_pair[0]][node_pair[1]]['{}_to_{}_port'.format(node_pair[0], node_pair[1])]['channels'][channel] - capacity_used_this_slot < 0:
                raise Exception('Tried to set up flow {} on edge {} channel {}, but this results in a negative channel capacity on this edge i.e. this edge\'s channel is full, cannot have more flow packets scheduled! Scheduler should not be giving invalid chosen flow sets to the environment.'.format(flow, node_pair, channel)) 

            # update edge capacity remaining after establish this flow
            self.network[node_pair[0]][node_pair[1]]['{}_to_{}_port'.format(node_pair[0], node_pair[1])]['channels'][channel] -= capacity_used_this_slot
            self.network[node_pair[0]][node_pair[1]]['{}_to_{}_port'.format(node_pair[0], node_pair[1])]['channels'][channel] = round(self.network[node_pair[0]][node_pair[1]]['{}_to_{}_port'.format(node_pair[0], node_pair[1])]['channels'][channel], num_decimals)

            if self.track_link_concurrent_demands_evolution:
                self.update_link_concurrent_demands_evolution(node_pair, num_concurrent_demands_to_add=1)

            # update global graph property
            self.network.graph['curr_nw_capacity_used'] += capacity_used_this_slot
        self.network.graph['num_active_connections'] += 1

        # update packets left for this flow
        self.update_flow_packets(flow)
        
    

    def take_down_connection(self, flow, num_decimals=6):
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
        packet_size = flow['packet_size']
        packets = flow['packets']
        packets_this_slot = flow['packets_this_slot']

        info_to_transfer_this_slot = packets_this_slot * packet_size
        capacity_used_this_slot = round(info_to_transfer_this_slot / self.slot_size, num_decimals) # info units of this flow transferred this time slot == capacity used on each channel in flow's path this time slot

        edges = self.get_path_edges(path)
        
        num_edges = len(edges)
        for edge in range(num_edges):
            node_pair = edges[edge]
            # update edge property
            # self.network[node_pair[0]][node_pair[1]]['channels'][channel] += capacity_used_this_slot
            # self.network[node_pair[0]][node_pair[1]]['channels'][channel] = round(self.network[node_pair[0]][node_pair[1]]['channels'][channel], num_decimals)
            self.network[node_pair[0]][node_pair[1]]['{}_to_{}_port'.format(node_pair[0], node_pair[1])]['channels'][channel] += capacity_used_this_slot
            self.network[node_pair[0]][node_pair[1]]['{}_to_{}_port'.format(node_pair[0], node_pair[1])]['channels'][channel] = round(self.network[node_pair[0]][node_pair[1]]['channels'][channel], num_decimals)

            # update global graph property
            self.network.graph['curr_nw_capacity_used'] -= capacity_used_this_slot
        self.network.graph['num_active_connections'] -= 1


    def path_cost(self, graph, path, weight=None):
        '''
        Calculates cost of path. If no weight specified, 1 unit of cost is 1
        link/edge in the path

        Args:
        - path (list): list of node labels making up path from src to dst
        - weight (dict key): label of weight to be considered when evaluating
        path cost

        Returns:
        - pathcost (int, float): total cost of path
        '''
        pathcost = 0
        
        for i in range(len(path)):
            if i > 0:
                edge = (path[i-1], path[i])
                if weight != None:
                    pathcost += 1
                    # bugged: if in future want 1 edge cost != 1, fix this
                    #pathcost += graph[edge[0]][edge[1]][weight]
                else:
                    # just count the number of edges
                    pathcost += 1

        return pathcost

    def k_shortest_paths(self, graph, source, target, num_k=None, weight='weight'):
        '''
        Uses Yen's algorithm to compute the k-shortest paths between a source
        and a target node. The shortest path is that with the lowest pathcost,
        defined by external path_cost() function. Paths are returned in order
        of path cost, with lowest past cost being first etc.

        Args:
        - source (label): label of source node
        - target (label): label of destination node
        - num_k (int, float): number of shortest paths to compute
        - weight (dict key): dictionary key of value to be 'minimised' when
        finding 'shortest' paths

        Returns:
        - A (list of lists): list of shortest paths between src and dst
        '''
        if num_k is None:
            num_k = self.num_k_paths

        # Shortest path from the source to the target
        A = [nx.shortest_path(graph, source, target, weight=weight)]
        A_costs = [self.path_cost(graph, A[0], weight)]

        # Initialize the heap to store the potential kth shortest path
        B = queue.PriorityQueue()

        for k in range(1, num_k):
            # spur node ranges first node to next to last node in shortest path
            try:
                for i in range(len(A[k-1])-1):
                    # Spur node retrieved from the prev k-shortest path, k - 1
                    spurNode = A[k-1][i]
                    # seq of nodes from src to spur node of prev k-shrtest path
                    rootPath = A[k-1][:i]

                    # We store the removed edges
                    removed_edges = []

                    for path in A:
                        if len(path) - 1 > i and rootPath == path[:i]:
                            # Remove edges of prev shrtest path w/ same root
                            edge = (path[i], path[i+1])
                            if not graph.has_edge(*edge):
                                continue
                            removed_edges.append((edge, graph.get_edge_data(*edge)))
                            graph.remove_edge(*edge)

                    # Calculate the spur path from the spur node to the sink
                    try:
                        spurPath = nx.shortest_path(graph, spurNode, target, weight=weight)

                        # Entire path is made up of the root path and spur path
                        totalPath = rootPath + spurPath
                        totalPathCost = self.path_cost(graph, totalPath, weight)
                        # Add the potential k-shortest path to the heap
                        B.put((totalPathCost, totalPath))

                    except nx.NetworkXNoPath:
                        pass

                    #Add back the edges that were removed from the graph
                    for removed_edge in removed_edges:
                        graph.add_edge(
                            *removed_edge[0],
                            **removed_edge[1]
                        )

                # Sort the potential k-shortest paths by cost
                # B is already sorted
                # Add the lowest cost path becomes the k-shortest path.
                while True:
                    try:
                        cost_, path_ = B.get(False)
                        if path_ not in A:
                            A.append(path_)
                            A_costs.append(cost_)
                            break
                    except queue.Empty:
                        break
            except IndexError:
                pass
        
        

        return A 

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
            name = 'sim_jobcentric_{}'.format(str(self.num_demands), str(self.job_centric))
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
        self.num_endpoints, self.num_pairs, self.endpoint_to_index, self.index_to_endpoint = tools.get_network_params(env.network.graph['endpoints'])
        self.ep_label = env.network.graph['endpoint_label']
        
        # init onehot encoding
        self.onehot_endpoints, self.endpoint_to_onehot = self.onehot_encode_endpoints()
        self.onehot_paths, self.path_to_onehot = self.onehot_encode_paths()
        self.num_paths = len(self.onehot_paths[0]) 

        # get init representation to get action embedding size
        init_rep, _ = self.gen_machine_readable_network_observation(self.env.network, return_onehot_vectors=False, dtype=tf.float16)
        self.num_actions = tf.shape(init_rep)[0]
        self.action_embedding_size = tf.shape(init_rep)[1]
        
    def onehot_encode_endpoints(self):
        onehot_endpoints = tf.one_hot(indices=list(self.index_to_endpoint.keys()), depth=self.num_endpoints)
        endpoint_to_onehot = {endpoint: onehot for endpoint, onehot in zip(self.index_to_endpoint.values(), onehot_endpoints)}
        
        return onehot_endpoints, endpoint_to_onehot
    
    def onehot_encode_paths(self):
        num_k_paths = self.env.num_k_paths
        all_paths = []
        for src in self.endpoint_to_index.keys():
            for dst in self.endpoint_to_index.keys():
                if src == dst:
                    pass
                else:
                    paths = self.env.k_shortest_paths(self.env.network, src, dst)
                    for path in paths:
                        if path[::-1] not in all_paths and path not in all_paths:
                            all_paths.append(path)
        indices = [i for i in range(len(all_paths))]
        self.path_to_index = {json.dumps(path): index for path, index in zip(all_paths, indices)}
        self.index_to_path = {index: json.dumps(path) for index, path in zip(indices, all_paths)}
        onehot_paths = tf.one_hot(indices=list(self.index_to_path.keys()), depth=len(all_paths))
        path_to_onehot = {path: onehot for path, onehot in zip(self.index_to_path.values(), onehot_paths)}
        
        return onehot_paths, path_to_onehot        

    def conv_human_readable_flow_to_machine_readable_flow(self, flow, return_onehot_vectors=False, dtype=tf.float32):
        machine_readable_flow = {}

        if return_onehot_vectors:
            # return onehot vector
            machine_readable_flow['src'] = tf.cast(self.endpoint_to_onehot[flow['src']], dtype=dtype)
            machine_readable_flow['dst'] = tf.cast(self.endpoint_to_onehot[flow['dst']], dtype=dtype)
        else:
            # return int
            machine_readable_flow['src'] = tf.cast(self.endpoint_to_index[flow['src']], dtype=dtype)
            machine_readable_flow['dst'] = tf.cast(self.endpoint_to_index[flow['dst']], dtype=dtype)
        try:
            if return_onehot_vectors:
                machine_readable_flow['path'] = tf.cast(self.path_to_onehot[json.dumps(flow['path'])], dtype=dtype)
            else:
                machine_readable_flow['path'] = tf.cast(self.path_to_index[json.dumps(flow['path'])], dtype=dtype)
        except KeyError:
            if return_onehot_vectors:
                machine_readable_flow['path'] = tf.cast(self.path_to_onehot[json.dumps(flow['path'][::-1])], dtype=dtype)
            else:
                machine_readable_flow['path'] = tf.cast(self.path_to_index[json.dumps(flow['path'][::-1])], dtype=dtype)
        machine_readable_flow['size'] = tf.cast(flow['size'], dtype=dtype)
        if flow['packets'] is None:
            machine_readable_flow['packets'] = tf.cast(-1, dtype=dtype)
        else:
            # machine_readable_flow['packets'] = tf.cast(len(flow['packets']), dtype=dtype)
            machine_readable_flow['packets'] = tf.cast(flow['packets'], dtype=dtype)
        machine_readable_flow['time_arrived'] = tf.cast(flow['time_arrived'], dtype=dtype)
        machine_readable_flow['selected'] = tf.cast(0, dtype=dtype)
        machine_readable_flow['null_action'] = tf.cast(0, dtype=dtype)
        machine_readable_flow['flow_present'] = tf.cast(1, dtype=dtype)

        return machine_readable_flow
    
    def gen_machine_readable_network_observation(self, network_observation, return_onehot_vectors=False, dtype=tf.float32):
        '''
        If return_onehot_vectors is False, rather than returning one hot encodings,
        will return discrete indices of variables. This is useful for gym.spaces.Discrete()
        observation spaces which automatically one-hot encode Discrete observation space variables.
        '''
        self.time_gen_machine_readable_start = time.time()

        num_placeholder_flows = self.num_endpoints * self.env.max_flows * (self.num_endpoints - 1)
        num_actions = num_placeholder_flows * self.env.num_k_paths
        
        # init action representations with empty (no) flows
        action_indices = [i for i in range(num_actions)]
        if return_onehot_vectors:
            # any discrete vars should be onehot encoded
            action_dict = {index: {'src': tf.cast(tf.zeros(len(self.onehot_endpoints[0])), dtype=dtype),
                                   'dst': tf.cast(tf.zeros(len(self.onehot_endpoints[0])), dtype=dtype),
                                   'path': tf.cast(tf.zeros(len(self.onehot_paths[0])), dtype=dtype),
                                   'size': tf.cast(-1, dtype=dtype),
                                   'packets': tf.cast(-1, dtype=dtype),
                                   'time_arrived': tf.cast(-1, dtype=dtype),
                                   'selected': tf.cast(0, dtype=dtype),
                                   'null_action': tf.cast(1, dtype=dtype),
                                   'flow_present': tf.cast(0, dtype=dtype)} for index in action_indices}
        else:
            # leave discrete vars as discrete ints
            action_dict = {index: {'src': tf.cast(-1, dtype=dtype),
                                   'dst': tf.cast(-1, dtype=dtype),
                                   'path': tf.cast(-1, dtype=dtype),
                                   'size': tf.cast(-1, dtype=dtype),
                                   'packets': tf.cast(-1, dtype=dtype),
                                   'time_arrived': tf.cast(-1, dtype=dtype),
                                   'selected': tf.cast(0, dtype=dtype),
                                   'null_action': tf.cast(1, dtype=dtype),
                                   'flow_present': tf.cast(0, dtype=dtype)} for index in action_indices}
        
        # go through network_observation queued flows and update action_dict representations
        action_iterator = iter(action_indices)
        for node in network_observation.nodes:
            if node[:len(self.ep_label)] == self.ep_label:
                queues = network_observation.nodes[node]
                for q in queues.values():
                    for i in range(self.env.max_flows):
                        idx = next(action_iterator)
                        try:
                            flow = q['queued_flows'][i]
                            paths = self.env.k_shortest_paths(self.env.network, flow['src'], flow['dst'])
                            for path in paths:
                                # each path is a separate action
                                # idx = next(action_iterator)
                                flow['path'] = path 
                                action_dict[idx] = self.conv_human_readable_flow_to_machine_readable_flow(flow, return_onehot_vectors, dtype)
                        except IndexError:
                            # no more flows in this queue 
                            pass
            else:
                # not an endpoint node
                pass
            
        # concat each action_dict key's values into single action_vector and stack all action vectors into single tensor
        action_stack = []
        for action in action_dict.keys():
            action_vector = self._stack_list_of_tensors([s for s in action_dict[action].values()], dtype=dtype)
            action_stack.append(action_vector)
        machine_readable_observation = tf.cast(action_stack, dtype=dtype)

        self.time_gen_machine_readable_end = time.time()
        
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








