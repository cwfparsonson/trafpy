from trafpy.generator.src import builder 
from trafpy.generator.src.dists import node_dists
from trafpy.generator.src.dists import val_dists
from trafpy.generator.src.dists import plot_dists
from trafpy.generator.src import flowcentric
from trafpy.generator.src import tools

import inspect
import sys
from tabulate import tabulate
import pandas as pd
import json
from statistics import mean
from itertools import chain
from collections import defaultdict # use for initialising arbitrary length nested dict
import numpy as np
import matplotlib.pyplot as plt

class Demand:
    def __init__(self,
                 demand_data,
                 eps,
                 name='demand'):

        self.eps = eps
        self.name = name
        self.reset(demand_data)

    def reset(self, demand_data):
        self.demand_data = demand_data
        self.num_demands = self.get_num_demands(self.demand_data)

        if 'job_id' in demand_data:
            self.job_centric = True
        else:
            self.job_centric = False

        self.num_control_deps, self.num_data_deps, self.num_flows = self.get_num_deps(demand_data)
        self.analyser = DemandAnalyser(self)

    def get_slots_dict(self, slot_size, print_info=False):
        return builder.construct_demand_slots_dict(demand_data=self.demand_data,
                                                   slot_size=slot_size,
                                                   print_info=print_info)

    def get_num_demands(self, demand_data):
        return len(demand_data['flow_id']) 

    def get_num_deps(self, demand_data):
        num_control_deps,num_data_deps,num_flows = 0, 0, 0

        if self.job_centric:
            # calc deps
            for job in demand_data['job']:
                for op in job.nodes:
                    flows = job.out_edges(op)
                    for flow in flows:
                        flow_stats = job.get_edge_data(flow[0],flow[1])
                        src = job.nodes[flow[0]]['attr_dict']['machine']
                        dst = job.nodes[flow[1]]['attr_dict']['machine']
                        if flow_stats['attr_dict']['dependency_type'] == 'data_dep':
                            num_data_deps+=1
                            if src != dst:
                                num_flows+=1
                        else:
                            num_control_deps+=1

        else:
            # 1 demand == 1 flow, therefore no dependencies & each demand == flow
            num_flows = self.num_demands
        
        return num_control_deps, num_data_deps, num_flows





class DemandAnalyser:
    
    def __init__(self, demand, subject_class_name=None):
        self.demand = demand
        if subject_class_name is None:
            self.subject_class_name = demand.name
        else:
            self.subject_class_name = subject_class_name
        self.computed_metrics = False

    def compute_metrics(self, print_summary=False):
        self.computed_metrics = True
        self._compute_general_summary()
        if self.demand.job_centric:
            self._compute_job_summary()
        else:
            self._compute_flow_summary()

        if print_summary:
            print('\n\n-=-=-=-=-=-= Summary =-=-=-=-=-=-=-')
            self._print_general_summary()
            if self.demand.job_centric:
                self._print_job_summary()
            else:
                self._print_flow_summary()



    ################################## GENERAL ################################################
    def _print_general_summary(self):
        print('\n~* General Information *~')
        print('Demand name: \'{}\''.format(self.demand.name))
        if self.demand.job_centric:
            print('Traffic type: Job-centric')
        else:
            print('Traffic type: Flow-centric')
        print('Total number of demands: {}'.format(self.num_demands))
        print('Time first demand arrived: {}'.format(self.time_first_demand_arrived))
        print('Time last demand arrived: {}'.format(self.time_last_demand_arrived))
        print('Total demand session duration: {}'.format(self.total_demand_session_duration))
        


    def _compute_general_summary(self):
        self.num_demands = self.demand.num_demands
        self.time_first_demand_arrived = min(self.demand.demand_data['event_time'])
        self.time_last_demand_arrived = max(self.demand.demand_data['event_time'])
        self.total_demand_session_duration = self.time_last_demand_arrived - self.time_first_demand_arrived


    ################################## FLOW ################################################
    def _print_flow_summary(self):
        print('\n~* Flow Information *~')
        print('Total number of flows: {}'.format(self.num_flows))
        print('Total flow info arrived: {}'.format(self.total_flow_info_arrived))
        print('Load rate (info units arrived per unit time): {}'.format(self.load_rate))
        print('Smallest flow size: {}'.format(self.smallest_flow_size))
        print('Largest flow size: {}'.format(self.largest_flow_size))



    def _compute_flow_summary(self):
        self.num_flows = self.demand.num_flows
        self.total_flow_info_arrived = sum(self.demand.demand_data['flow_size'])
        self.load_rate = flowcentric.get_flow_centric_demand_data_overall_load_rate(self.demand.demand_data)
        self.smallest_flow_size = min(self.demand.demand_data['flow_size'])
        self.largest_flow_size = max(self.demand.demand_data['flow_size'])



    ################################## JOB ################################################
    def _print_job_summary(self):
        print('\n~* Job Information *~')
        print('Total number of control dependencies: {}'.format(self.num_control_deps))
        print('Total number of data dependencies: {}'.format(self.num_data_deps))


    def _compute_job_summary(self):
        self.num_control_deps = self.demand.num_control_deps
        self.num_data_deps = self.demand.num_data_deps



class DemandsAnalyser:
    def __init__(self, *demands):
        self.demands = demands
        self.computed_metrics = False

    def _check_analyser_valid(self, analyser):
        if inspect.isclass(analyser):
            raise Exception('Must instantiate DemandAnalyser class before passing to DemandPlotter.')

        if not analyser.computed_metrics:
            raise Exception('Must compute metrics with DemandAnalyser.compute_metrics() before passing to DemandPlotter.')

    def compute_metrics(self, print_summary=False):
        self.computed_metrics = True

        if self.demands[0].job_centric:
            self._compute_job_summary()
        else:
            self._compute_flow_summary()

        if print_summary:
            df = pd.DataFrame(self.summary_dict)
            print(tabulate(df, headers='keys', tablefmt='psql'))

    def _compute_flow_summary(self):
        self.summary_dict = {'Name': [], 
                             'Flows': [], 
                             '1st': [], 
                             'Last': [],
                             'Duration': [],
                             'Info': [],
                             'Load': [],
                             'Smallest': [],
                             'Largest': []}
        for demand in self.demands:
            demand.analyser.compute_metrics(print_summary=False)
            self.summary_dict['Name'].append(demand.analyser.subject_class_name)
            self.summary_dict['Flows'].append(demand.analyser.num_flows)
            self.summary_dict['1st'].append(demand.analyser.time_first_demand_arrived)
            self.summary_dict['Last'].append(demand.analyser.time_last_demand_arrived)
            self.summary_dict['Duration'].append(demand.analyser.total_demand_session_duration)
            self.summary_dict['Info'].append(demand.analyser.total_flow_info_arrived)
            self.summary_dict['Load'].append(demand.analyser.load_rate)
            self.summary_dict['Smallest'].append(demand.analyser.smallest_flow_size)
            self.summary_dict['Largest'].append(demand.analyser.largest_flow_size)

    def _compute_job_summary(self):
        raise NotImplementedError





class DemandPlotter:
    def __init__(self, demand):
        self.demand = demand

    def _check_analyser_valid(self, analyser):
        if inspect.isclass(analyser):
            raise Exception('Must instantiate DemandAnalyser class before passing to DemandPlotter.')

        if not analyser.computed_metrics:
            raise Exception('Must compute metrics with DemandAnalyser.compute_metrics() before passing to DemandPlotter.')

    def plot_flow_size_dist(self, logscale=True, num_bins=20):
        return plot_dists.plot_val_dist(self.demand.demand_data['flow_size'], show_fig=False, logscale=logscale, num_bins=num_bins, rand_var_name='Flow Size')

    def plot_interarrival_time_dist(self, logscale=True, num_bins=20):
        interarrival_times = [self.demand.demand_data['event_time'][i+1]-self.demand.demand_data['event_time'][i] for i in range(self.demand.num_demands-1)]
        return plot_dists.plot_val_dist(interarrival_times, show_fig=False, logscale=logscale, num_bins=num_bins, rand_var_name='Interarrival Time')

    def plot_node_load_dists(self, eps, ep_link_bandwidth=None):
        '''
        1. Returns bar chart of end point links on x-axis and corresponding load on
        y-axis. If ep_link_bandwidth not given, y-axis will be absolute info units
        per unit time (load rate). If given, y-axis will be load (fraction).

        2. Returns same bar chart but y-axis is fraction of overall network load
        being requested by each end point link.

        3. (if ep_link_bandwidth not None) Returns same bar chart by y-axis is 
        fraction of overall network capacity being requested by each end point link
        '''
        index_to_pair, pair_to_index = node_dists.get_network_pair_mapper(eps)
        _, _, node_to_index, index_to_node = tools.get_network_params(eps)
        ep_loads = {node_to_index[ep]: None for ep in eps}
        figs = []
        
        fig1 = plt.figure()
        for ep in ep_loads.keys():
            ep_loads[ep] = flowcentric.get_flow_centric_demand_data_ep_load_rate(self.demand.demand_data, index_to_node[ep], eps)
            if ep_link_bandwidth is not None:
                ep_loads[ep] /= ep_link_bandwidth

        if ep_link_bandwidth is None:
            ylabel = 'End Point Load (Absolute)'
            ylim = None
        else:
            ylabel = 'End Point Load (Fraction)'
            ylim = None
        xlabel = 'End Point Link'
        plot_dists.plot_val_bar(ep_loads.keys(), ep_loads.values(), ylabel, ylim, xlabel, show_fig=False)
        figs.append(fig1)

        fig2 = plt.figure()
        overall_load_rate = flowcentric.get_flow_centric_demand_data_overall_load_rate(self.demand.demand_data)
        ep_loads_as_frac_of_overall_load = {}
        for ep in ep_loads.keys():
            ep_loads_as_frac_of_overall_load[ep] = flowcentric.get_flow_centric_demand_data_ep_load_rate(self.demand.demand_data, index_to_node[ep], eps)
            ep_loads_as_frac_of_overall_load[ep] /= overall_load_rate
        ylabel = 'Fraction of Overall Load Requested'
        plot_dists.plot_val_bar(ep_loads_as_frac_of_overall_load.keys(), ep_loads_as_frac_of_overall_load.values(), ylabel, ylim, xlabel, show_fig=False)
        figs.append(fig2)

        if ep_link_bandwidth is not None:
            fig3 = plt.figure()
            overall_network_capacity = len(eps) * ep_link_bandwidth
            ep_loads_as_frac_of_overall_capacity = {}
            for ep in ep_loads.keys():
                ep_loads_as_frac_of_overall_capacity[ep] = flowcentric.get_flow_centric_demand_data_ep_load_rate(self.demand.demand_data, index_to_node[ep], eps)
                ep_loads_as_frac_of_overall_capacity[ep] /=overall_network_capacity 
            ylabel = 'Fraction of Overall Capacity Requested'
            plot_dists.plot_val_bar(ep_loads_as_frac_of_overall_capacity.keys(), ep_loads_as_frac_of_overall_capacity.values(), ylabel, ylim, xlabel, show_fig=False)
            figs.append(fig3)


        return figs
        






    def plot_node_load_fraction_of_overall_load_dist(self, eps, network_rate_capacity):
        '''
        Returns bar chart of end point links on x-axis and corresponding fraction
        of the overall network capacity

        '''



    def plot_node_dist(self, eps):

        sampled_pairs = {}
        sources = self.demand.demand_data['sn']
        destinations = self.demand.demand_data['dn']
        for i in range(self.demand.num_demands):
            sn = sources[i]
            dn = destinations[i]
            pair = json.dumps([sn, dn])
            if pair not in sampled_pairs:
                # check if switched src-dst pair that already occurred
                pair_switched = json.dumps([dn, sn])
                if pair_switched not in sampled_pairs:
                    sampled_pairs[pair] = 1 # init first occurrence of pair
                else:
                    # pair already seen before
                    sampled_pairs[pair_switched] += 1
            else:
                # pair already seen before
                sampled_pairs[pair] += 1

        node_dist = node_dists.convert_sampled_pairs_into_node_dist(sampled_pairs, eps)

        return plot_dists.plot_node_dist(node_dist)

    def find_index_of_int_in_str(self, string):
        idx = 0
        for char in string:
            try:
                int(char)
                return idx
            except ValueError:
                # char is not an int
                idx += 1
        raise Exception('Could not find an integer in the string {}'.format(string))

    def plot_link_loads_vs_time(self, net, slot_size, demand, mean_period=None, logscale=False):
        # slot_size == time period over which to consider a flow as requesting resources
        slots_dict = demand.get_slots_dict(slot_size)
        
        ep_links = []
        for link in net.edges:
            if net.graph['endpoint_label'] not in json.dumps(link):
                # not an endpoint link
                pass
            else:
                ep_links.append(link)
        ep_to_link = {}
        for ep_link in ep_links:
            for ep in ep_link:
                if net.graph['endpoint_label'] in ep:
                    ep_to_link[ep] = json.dumps(ep_link)

        link_load_dict = {json.dumps([link[0],link[1]]):
                                    {'time_slots': [t for t in slots_dict.keys()],
                                     'loads_abs': [0 for _ in range(len(slots_dict.keys()))]}
                                    for link in ep_links}

        time_slots = iter(list(slots_dict.keys()))
        for idx in range(len(slots_dict.keys())):
            for flow in slots_dict[next(time_slots)]['new_event_dicts']:
                bw_requested_this_slot = flow['size'] / slot_size
                sn, dn = flow['src'], flow['dst']
                # update load for src and dst ep links
                link_load_dict[ep_to_link[sn]]['loads_abs'][idx] += bw_requested_this_slot
                link_load_dict[ep_to_link[dn]]['loads_abs'][idx] += bw_requested_this_slot

        nested_dict = lambda: defaultdict(nested_dict)
        plot_dict1 = nested_dict()
        plot_dict2 = nested_dict()
        for link in link_load_dict.keys():
            time_slots = link_load_dict[link]['time_slots']
            loads_abs = link_load_dict[link]['loads_abs']

            # average every n elements (time slots) in lists to smooth line plot
            n = mean_period
            if n == 'all':
                # avrg over all time slots
                n = time_slots[-1]
                avrg_load = np.mean(np.array(loads_abs))
                loads_abs = [avrg_load for _ in range(len(time_slots))]
            elif n is not None:
                time_slots = list(chain.from_iterable([mean(time_slots[i:i+n])]*n for i in range(0,len(time_slots),n)))
                loads_abs = list(chain.from_iterable([mean(loads_abs[i:i+n])]*n for i in range(0,len(loads_abs),n)))
            else:
                # not smoothing results by taking mean
                pass
            plot_dict1[link]['x_values'] = time_slots
            plot_dict1[link]['y_values'] = loads_abs 

            link_bw_capacity = net[json.loads(link)[0]][json.loads(link)[1]]['max_channel_capacity'] * net.graph['num_channels_per_link']
            loads_frac = [load_abs / link_bw_capacity for load_abs in loads_abs]
            plot_dict2[link]['x_values'] = time_slots
            plot_dict2[link]['y_values'] = loads_frac


        # load absolute
        fig1 = plot_dists.plot_val_line(plot_dict=plot_dict1, xlabel='Time Slot', ylabel='Link Load (Abs)', linewidth=0.4, alpha=1, show_fig=False)

        # load fraction
        fig2 = plot_dists.plot_val_line(plot_dict=plot_dict2, xlabel='Time Slot', ylabel='Link Load (Frac)', linewidth=0.4, alpha=1, show_fig=False)

        return [fig1, fig2]









# TODO
# flow_size_dist
# interarrival_time_dist
# node_dist


class DemandsPlotter:
    def __init__(self, *demands):
        self.demands = demands
        self.classes = self._group_analyser_classes(*self.demands)

    def _group_analyser_classes(self, *demands):
        classes = []
        for demand in demands:
            if demand.name not in classes:
                classes.append(demand.name)

        return classes

    def _check_analyser_valid(self, analyser):
        if inspect.isclass(analyser):
            raise Exception('Must instantiate DemandAnalyser class before passing to DemandPlotter.')

        if not analyser.computed_metrics:
            raise Exception('Must compute metrics with DemandAnalyser.compute_metrics() before passing to DemandPlotter.')

    def plot_flow_size_dists(self, logscale=False):
        plot_dict = {_class: {'rand_vars': []} for _class in self.classes}
        for demand in self.demands:
            plot_dict[demand.name]['rand_vars'] = demand.demand_data['flow_size']

        fig = plot_dists.plot_multiple_kdes(plot_dict, plot_hist=False, xlabel='Flow Size', ylabel='Density', logscale=logscale, show_fig=False)

        return fig

    def _get_demand_interarrival_times(self, demand):
        return [demand.demand_data['event_time'][i+1]-demand.demand_data['event_time'][i] for i in range(demand.num_demands-1)]

    def plot_interarrival_time_dists(self, logscale=False):
        plot_dict = {_class: {'rand_vars': []} for _class in self.classes}
        for demand in self.demands:
            plot_dict[demand.name]['rand_vars'] = self._get_demand_interarrival_times(demand)

        fig = plot_dists.plot_multiple_kdes(plot_dict, plot_hist=False, xlabel='Interarrival Time', ylabel='Density', logscale=logscale, show_fig=False)

        return fig



        




























