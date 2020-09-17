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


class SRPT(SchedulerToolbox):

    def __init__(self):
        super().__init__()
        self.scheduler_name = 'srpt'

    def get_scheduler_action(self, observation):
        '''
        Uses observation and chosen rwa action(s) to construct schedule for this
        timeslot
        '''
        # update scheduler network & new flow states
        self.update_network_state(observation, hide_child_dependency_flows=True)

        # choose which flows to schedule for this time slot
        chosen_flows = []
        for ep in self.SchedulerNetwork.graph['endpoints']:
            queues = self.SchedulerNetwork.nodes[ep]
            for queue in queues.keys():
                queued_flows = queues[queue]['queued_flows']
                completion_times = queues[queue]['completion_times']
                num_queued_flows = len(queues[queue]['queued_flows'])
                if num_queued_flows == 0:
                    # no flows queued, move to next queue
                    continue
                else:
                    # queued flows present
                    chosen_flow, _ = self.find_shortest_flow_in_queue(queued_flows,completion_times)
                   
                    # check for contentions
                    contending_flow = None
                    establish_flow = False
                    if len(chosen_flows) != 0:
                        establish_flow, p, c = self.look_for_available_lightpath(chosen_flow,chosen_flows)
                        chosen_flow['path'], chosen_flow['channel'] = p, c
                        if not establish_flow:
                            contending_flow,contending_flow_fct,p,c = self.find_contending_flow(chosen_flow,chosen_flows)
                            chosen_flow['path'], chosen_flow['channel'] = p, c
                            comp_time, _ = self.estimate_time_to_completion(chosen_flow)

                            if contending_flow_fct > comp_time:
                                # new choice has lower fct that established flow
                                establish_flow = True
                            else:
                                # established flow has lower fct, do not choose
                                pass
                        else:
                            # rwa was completed
                            pass
                    else:
                        # no existing chosen flows yet, can choose flow
                        establish_flow = True
                    
                    if establish_flow:
                        try:
                            chosen_flows.remove(contending_flow)
                        except (NameError, ValueError):
                            # already not present
                            pass
                        chosen_flows.append(chosen_flow)
                    else:
                        # contention was found and lost
                        pass
                            

        return chosen_flows

    def get_action(self, observation):

        # scheduler action
        chosen_flows = self.get_scheduler_action(observation)
        action = {'chosen_flows': chosen_flows}

        return action
    
    def find_contending_flow(self, chosen_flow, chosen_flows):
        '''
        Goes through chosen flow possible path & channel combinations & 
        compares to path-channel combinations in chosen flows. Saves all
        contentions that arise. When all possible contentions have been checked,
        finds the 'least contentious' (i.e. longest flow completion time) in
        chosen_flows and returns this as the contending flow (since other
        flows in chosen_flows will have a lower FCT than least contentious flow
        and therefore should not be removed)
        '''
        
        contending_flows = {'contending_flows': []}
        paths = chosen_flow['k_shortest_paths']
        channels = self.RWA.channel_names
        
        path = chosen_flow['path']
        edges = self.get_path_edges(path)
        channel = chosen_flow['channel']

        taken_paths = [flow['path'] for flow in chosen_flows]
        taken_edges = [self.get_path_edges(taken_path) for taken_path in taken_paths]
        taken_channels = [flow['channel'] for flow in chosen_flows]
        found_contention=False
        contending_comp_times = []
        for path in paths:
            edges = self.get_path_edges(path)
            for channel in channels:
                for idx in range(len(taken_paths)):
                    taken_path_edges = taken_edges[idx]
                    taken_channel = taken_channels[idx]
                    if (channel == taken_channel and any(e in taken_path_edges for e in edges)) or (channel == taken_channel and any(e[::-1] in taken_path_edges for e in edges)):
                        found_contention=True
                        contending_flows['contending_flows'].append({'cont_f': chosen_flows[idx],
                                                                     'chosen_p': path,
                                                                     'chosen_c': channel})
                        t, _ = self.estimate_time_to_completion(chosen_flows[idx])
                        contending_comp_times.append(t)
        if found_contention == False:
            sys.exit('ERROR: Could not find where contention was')

        idx_max_fct = contending_comp_times.index(max(contending_comp_times))
        contending_flow = contending_flows['contending_flows'][idx_max_fct]['cont_f']
        contending_flow_fct = contending_comp_times[idx_max_fct]
        chosen_path = contending_flows['contending_flows'][idx_max_fct]['chosen_p']
        chosen_channel = contending_flows['contending_flows'][idx_max_fct]['chosen_c']
        
         
        return contending_flow, contending_flow_fct, chosen_path, chosen_channel
        

