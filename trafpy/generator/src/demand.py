from trafpy.generator.src import builder 
from trafpy.generator.src.dists import node_dists
from trafpy.generator.src.dists import val_dists
from trafpy.generator.src.dists import plot_dists

import inspect
import sys
from tabulate import tabulate
import pandas as pd
import json

class Demand:
    def __init__(self,
                 demand_data,
                 name='demand'):

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
        if 0 in demand_data['establish']:
            takedowns_present = True
        else:
            takedowns_present = False

        if takedowns_present:
            # half events are takedowns for demand establishments
            num_demands = int(len(demand_data['establish'])/2)
        else:
            # all events are new demands
            num_demands = int(len(demand_data['establish']))

        return num_demands

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
    
    def __init__(self, demand, bidirectional_links=True, subject_class_name=None):
        self.demand = demand
        if subject_class_name is None:
            self.subject_class_name = demand.name
        else:
            self.subject_class_name = subject_class_name
        self.computed_metrics = False

    def compute_metrics(self, bidirectional_links=True, print_summary=False):
        self.computed_metrics = True
        self.bidirectional_links = bidirectional_links
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
        if self.bidirectional_links:
            # 1 flow occupies 2 endpoint links therefore has 2*flow_size load
            self.load_rate = sum(2*self.demand.demand_data['flow_size'])/(max(self.demand.demand_data['event_time'])-min(self.demand.demand_data['event_time']))
        else:
            self.load_rate = sum(self.demand.demand_data['flow_size'])/(max(self.demand.demand_data['event_time'])-min(self.demand.demand_data['event_time']))
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

    def plot_node_dist(self, eps, logscale=True, num_bins=20):
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



        




























