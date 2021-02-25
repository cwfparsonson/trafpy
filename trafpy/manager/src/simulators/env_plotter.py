import inspect
from trafpy.generator.src.dists import plot_dists
import matplotlib.pyplot as plt
from collections import defaultdict # use for initialising arbitrary length nested dict
import json
from statistics import mean
from itertools import chain
from tabulate import tabulate
import pandas as pd
import numpy as np


class EnvPlotter:
    def __init__(self):
        pass


    def _check_analyser_valid(self, analyser):
        if inspect.isclass(analyser):
            raise Exception('Must instantiate EnvAnalyser class before passing to EnvPlotter.')

        if not analyser.computed_metrics:
            raise Exception('Must compute metrics with EnvAnalyser.compute_metrics() before passing to EnvPlotter.')



class EnvsPlotter:
    def __init__(self):
        pass

    def _group_analyser_classes(self, *analysers):
        classes = []
        for analyser in analysers:
            if analyser.subject_class_name not in classes:
                classes.append(analyser.subject_class_name)

        return classes

    def _check_analyser_valid(self, analyser):
        if inspect.isclass(analyser):
            raise Exception('Must instantiate EnvAnalyser class before passing to EnvPlotter.')

        if not analyser.computed_metrics:
            raise Exception('Must compute metrics with EnvAnalyser.compute_metrics() before passing to EnvPlotter.')



    ############################### GENERIC #################################
    def display_t_score_table(self, *analysers):
        _summary_dict = {'Load': [],
                        'Subject': [],
                        'T-Score': []}
        for analyser in analysers:
            self._check_analyser_valid(analyser)
            _summary_dict['Load'].append(round(analyser.load_frac, 2))
            _summary_dict['Subject'].append(analyser.subject_class_name)
            _summary_dict['T-Score'].append(analyser.t_score)

        # sort by order of load
        index = np.argsort(_summary_dict['Load'])
        summary_dict = {}
        for key in _summary_dict.keys():
            summary_dict[key] = np.asarray(_summary_dict[key])[index]
        
        dataframe = pd.DataFrame(summary_dict)
        print(tabulate(dataframe, headers='keys', tablefmt='psql'))

    def plot_t_score_scatter(self, *analysers):
        '''Plots performance indicators for T-scores.

        TODO:
            Currently plots FCT component vs. throughput component, but should add
            colour bar so can include dropped flow component (similar to final
            scatter in JLT plot).


        '''
        nested_dict = lambda: defaultdict(nested_dict)
        plot_dict = nested_dict()
        for analyser in analysers:
            self._check_analyser_valid(analyser)
            plot_dict[analyser.load_frac][analyser.subject_class_name]['x_values'] = analyser.throughput_component 
            plot_dict[analyser.load_frac][analyser.subject_class_name]['y_values'] = analyser.fct_component 


        figs = []
        for load in plot_dict.keys():
            fig = plot_dists.plot_val_scatter(plot_dict=plot_dict[load], xlabel='Load {} Throughput Component'.format(str(round(load,2))), ylabel='Load {} FCT Component'.format(str(round(load,2))), alpha=1.0, marker_size=60, logscale=True, show_fig=False)
            figs.append(fig)

        return figs

        

    def plot_average_fct_vs_load(self, *analysers):
        '''
        *analysers (*args): Analyser objects whose metrics you wish to plot.
        '''
        classes = self._group_analyser_classes(*analysers)
        plot_dict = {_class: {'x_values': [], 'y_values': [], 'rand_vars': []} for _class in classes}

        for analyser in analysers:
            self._check_analyser_valid(analyser)
            plot_dict[analyser.subject_class_name]['x_values'].append(analyser.load_frac)
            plot_dict[analyser.subject_class_name]['y_values'].append(analyser.average_fct)
            plot_dict[analyser.subject_class_name]['rand_vars'].append(analyser.average_fct)

        # scatter
        fig1 = plot_dists.plot_val_scatter(plot_dict=plot_dict, xlabel='Load', ylabel='Average FCT', show_fig=False)

        # complementary cdf
        fig2 = plot_dists.plot_val_cdf(plot_dict=plot_dict, xlabel='Average FCT', ylabel='Complementary CDF', complementary_cdf=True, show_fig=False)

        return [fig1, fig2]

    def plot_99th_percentile_fct_vs_load(self, *analysers):
        classes = self._group_analyser_classes(*analysers)
        plot_dict = {_class: {'x_values': [], 'y_values': [], 'rand_vars': []} for _class in classes}

        for analyser in analysers:
            self._check_analyser_valid(analyser)
            plot_dict[analyser.subject_class_name]['x_values'].append(analyser.load_frac)
            plot_dict[analyser.subject_class_name]['y_values'].append(analyser.nn_fct)
            plot_dict[analyser.subject_class_name]['rand_vars'].append(analyser.nn_fct)

        # scatter
        fig1 = plot_dists.plot_val_scatter(plot_dict=plot_dict, xlabel='Load', ylabel='99th Percentile FCT', show_fig=False)

        # complementary cdf
        fig2 = plot_dists.plot_val_cdf(plot_dict=plot_dict, xlabel='99th Percentile FCT', ylabel='Complementary CDF', complementary_cdf=True, show_fig=False)

        return [fig1, fig2]

    def plot_max_fct_vs_load(self, *analysers):
        classes = self._group_analyser_classes(*analysers)
        plot_dict = {_class: {'x_values': [], 'y_values': [], 'rand_vars': []} for _class in classes}

        for analyser in analysers:
            self._check_analyser_valid(analyser)
            plot_dict[analyser.subject_class_name]['x_values'].append(analyser.load_frac)
            plot_dict[analyser.subject_class_name]['y_values'].append(analyser.max_fct)
            plot_dict[analyser.subject_class_name]['rand_vars'].append(analyser.max_fct)

        # scatter
        fig1 = plot_dists.plot_val_scatter(plot_dict=plot_dict, xlabel='Load', ylabel='Max FCT', show_fig=False)

        # complementary cdf
        fig2 = plot_dists.plot_val_cdf(plot_dict=plot_dict, xlabel='Max FCT', ylabel='Complementary CDF', complementary_cdf=True, show_fig=False)

        return [fig1, fig2]

    def plot_fcts_cdf_for_different_loads(self, *analysers):
        # plot cdf of all fcts for each test subject for different loads
        classes = self._group_analyser_classes(*analysers)
        # plot_dict = {_class: {'x_values': [], 'rand_vars': []} for _class in classes}

        nested_dict = lambda: defaultdict(nested_dict)
        plot_dict = nested_dict()
        for analyser in analysers:
            self._check_analyser_valid(analyser)
            plot_dict[analyser.load_frac][analyser.subject_class_name]['rand_vars'] = analyser.fcts

        # complementary cdf
        figs = []
        for load in plot_dict.keys():
            fig = plot_dists.plot_val_cdf(plot_dict=plot_dict[load], xlabel='Load {} FCTs'.format(round(load,2)), ylabel='Complementary CDF', logscale=True, plot_points=False, complementary_cdf=True, show_fig=False)
            figs.append(fig)

        return figs

    def plot_link_fcts_cdf_for_different_loads(self, *analysers):
        pass



    def plot_fraction_of_arrived_flows_dropped_vs_load(self, *analysers):
        classes = self._group_analyser_classes(*analysers)
        plot_dict = {_class: {'x_values': [], 'y_values': [], 'rand_vars': []} for _class in classes}

        for analyser in analysers:
            self._check_analyser_valid(analyser)
            plot_dict[analyser.subject_class_name]['x_values'].append(analyser.load_frac)
            plot_dict[analyser.subject_class_name]['y_values'].append(analyser.dropped_flow_frac)
            plot_dict[analyser.subject_class_name]['rand_vars'].append(analyser.dropped_flow_frac)

        # scatter 
        fig1 = plot_dists.plot_val_scatter(plot_dict=plot_dict, xlabel='Load', ylabel='Flows Dropped', show_fig=False)

        # complementary cdf
        fig2 = plot_dists.plot_val_cdf(plot_dict=plot_dict, xlabel='Flows Dropped', ylabel='Complementary CDF', complementary_cdf=True, show_fig=False)

        return [fig1, fig2]



    def plot_src_dst_queue_evolution_for_different_loads(self, src, dst, length_type='queue_lengths_num_flows', *analysers):
        if length_type != 'queue_lengths_num_flows' and length_type != 'queue_lengths_info_units':
            raise Exception('length_type must be either \'queue_lengths_num_flows\' or \'queue_lengths_info_units\', but is {}'.format(length_type))
        classes = self._group_analyser_classes(*analysers)

        nested_dict = lambda: defaultdict(nested_dict)
        plot_dict = nested_dict()
        load_to_meas_time = {} # collect measurement times for verical line plotting
        for analyser in analysers:
            if not analyser.env.track_queue_evolution:
                raise Exception('Must set track_queue_evolution=True when instantiating env simulation in order to plot queue evolution.')

            self._check_analyser_valid(analyser)
            plot_dict[analyser.load_frac][analyser.subject_class_name]['x_values'] = analyser.env.queue_evolution_dict[src][dst]['times']
            plot_dict[analyser.load_frac][analyser.subject_class_name]['y_values'] = analyser.env.queue_evolution_dict[src][dst][length_type]
            load_to_meas_time[analyser.load_frac] = [analyser.measurement_start_time, analyser.measurement_end_time]

        # set y-axis limits
        if length_type == 'queue_lengths_num_flows':
            load_to_ylim = {}
            for analyser in analysers:
                if analyser.env.max_flows is not None:
                    load_to_ylim[analyser.load_frac] = [0, int(1.25*analyser.env.max_flows)]
                else:
                    # no max number of flows
                    load_to_ylim[analyser.load_frac] = None
        else:
            load_to_ylim = {analyser.load_frac: None for analyser in analysers}

        figs = []
        for load in plot_dict.keys():
            figs.append(plot_dists.plot_val_line(plot_dict=plot_dict[load], xlabel='Time', ylabel='Load {} {}-{} Queue Length'.format(str(round(load,2)), src, dst), ylim=load_to_ylim[load], vertical_lines=[load_to_meas_time[load][0], load_to_meas_time[load][1]], show_fig=False))
        
        return figs

    def plot_throughput_vs_load(self, *analysers):
        classes = self._group_analyser_classes(*analysers)

        # init plot dict
        nested_dict = lambda: defaultdict(nested_dict)
        plot_dict = nested_dict()
        plot_dict2 = {_class: {'x_values': [], 'y_values': [], 'rand_vars': []} for _class in classes}
        load_to_meas_time = {} # collect measurement times for verical line plotting
        for analyser in analysers:
            self._check_analyser_valid(analyser)
            plot_dict[analyser.load_frac]['x_values'] = []
            plot_dict[analyser.load_frac]['y_values'] = []


        for analyser in analysers:
            plot_dict[analyser.load_frac]['x_values'].append(analyser.subject_class_name)
            plot_dict[analyser.load_frac]['y_values'].append(analyser.throughput_abs)

            plot_dict2[analyser.subject_class_name]['x_values'].append(analyser.load_frac)
            plot_dict2[analyser.subject_class_name]['y_values'].append(analyser.throughput_abs)
            plot_dict2[analyser.subject_class_name]['rand_vars'].append(analyser.throughput_abs)

        # individual bar chars
        figs = []
        for load in plot_dict.keys():
            figs.append(plot_dists.plot_val_bar(x_values=plot_dict[load]['x_values'], y_values=plot_dict[load]['y_values'], xlabel='Test Subject Class', ylabel='Load {} Throughput Rate'.format(str(round(load,2)), show_fig=False)))

        # scatter
        fig1 = plot_dists.plot_val_scatter(plot_dict=plot_dict2, xlabel='Load', ylabel='Throughput Rate', show_fig=False)

        # complementary cdf
        fig2 = plot_dists.plot_val_cdf(plot_dict=plot_dict2, xlabel='Throughput Rate', ylabel='Complementary CDF', complementary_cdf=True, show_fig=False)

        return [figs, fig1, fig2]




    def plot_demand_slot_colour_grid_for_different_schedulers(self, *analysers):
        figs = []
        for analyser in analysers:
            if not analyser.env.track_grid_slot_evolution:
                raise Exception('Must set track_grid_slot_evolution=True when instantiating env simulation in order to plot grid.')
            self._check_analyser_valid(analyser)
            figs.append(plot_dists.plot_demand_slot_colour_grid(analyser.grid_demands, title=analyser.env.sim_name, xlim=None, show_fig=False))

        return figs

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

    def plot_link_utilisation_vs_time_for_different_loads(self, *analysers, **kwargs):
        nested_dict = lambda: defaultdict(nested_dict)
        plot_dict = nested_dict()

        link_types = []
        for analyser in analysers:
            if not analyser.env.track_link_utilisation_evolution:
                raise Exception('Must set track_link_utilisation_evolution=True when instantiating env simulation in order to plot link utilisation.')
            self._check_analyser_valid(analyser)

            for link in analyser.env.link_utilisation_dict.keys():
                # assume node labels have format of edge type followed by underscore and int e.g. 'server_1', 'core_3' etc
                node1, node2 = json.loads(link)[0], json.loads(link)[1]
                idx1, idx2 = self.find_index_of_int_in_str(node1), self.find_index_of_int_in_str(node2)
                link_type = [node1[:idx1-1],node2[:idx2-1]]
                flip_link=False
                if link_type not in link_types:
                    flip_link=True
                    if link_type[::-1] not in link_types:
                        flip_link=False
                        link_types.append(link_type)
                if flip_link:
                    link_type = link_type[::-1]

                time_slots = analyser.env.link_utilisation_dict[link]['time_slots']
                util = analyser.env.link_utilisation_dict[link]['util']

                # average every n elements (time slots) in lists to smooth line plot
                n = kwargs['mean_period']
                if 'mean_period' in kwargs.keys():
                    n = kwargs['mean_period']
                    time_slots = list(chain.from_iterable([mean(time_slots[i:i+n])]*n for i in range(0,len(time_slots),n)))
                    util = list(chain.from_iterable([mean(util[i:i+n])]*n for i in range(0,len(util),n)))
                else:
                    # not smoothing results by taking mean
                    pass

                plot_dict[analyser.subject_class_name][json.dumps(link_type)][analyser.load_frac][link]['x_values'] = time_slots
                plot_dict[analyser.subject_class_name][json.dumps(link_type)][analyser.load_frac][link]['y_values'] = util
                    

        load_to_meas_time = {} # collect measurement times for verical line plotting
        for analyser in analysers:
            self._check_analyser_valid(analyser)
            load_to_meas_time[analyser.load_frac] = [int(analyser.measurement_start_time/analyser.env.slot_size), int(analyser.measurement_end_time/analyser.env.slot_size)]


        figs = []
        for subject_class in plot_dict.keys():
            for link_type in plot_dict[subject_class].keys():
                for load in plot_dict[subject_class][link_type].keys():
                    fig = plot_dists.plot_val_line(plot_dict=plot_dict[subject_class][link_type][load], xlabel='Time Slot', ylabel='{} Load {} Link Util'.format(subject_class, str(round(load,2))), ylim=[0,1], linewidth=0.4, alpha=1, vertical_lines=load_to_meas_time[load], show_fig=False)
                    figs.append(fig)

        return figs


    def plot_link_concurrent_demands_vs_time_for_different_loads(self, *analysers, **kwargs):
        nested_dict = lambda: defaultdict(nested_dict)
        plot_dict = nested_dict()

        link_types = []
        for analyser in analysers:
            if not analyser.env.track_link_concurrent_demands_evolution:
                raise Exception('Must set track_link_concurrent_demands_evolution=True when instantiating env simulation in order to plot number of concurrent demands.')
            self._check_analyser_valid(analyser)

            for link in analyser.env.link_concurrent_demands_dict.keys():
                # assume node labels have format of edge type followed by underscore and int e.g. 'server_1', 'core_3' etc
                node1, node2 = json.loads(link)[0], json.loads(link)[1]
                idx1, idx2 = self.find_index_of_int_in_str(node1), self.find_index_of_int_in_str(node2)
                link_type = [node1[:idx1-1],node2[:idx2-1]]
                flip_link=False
                if link_type not in link_types:
                    flip_link=True
                    if link_type[::-1] not in link_types:
                        flip_link=False
                        link_types.append(link_type)
                if flip_link:
                    link_type = link_type[::-1]

                time_slots = analyser.env.link_concurrent_demands_dict[link]['time_slots']
                concurrent_demands = analyser.env.link_concurrent_demands_dict[link]['concurrent_demands']

                # average every n elements (time slots) in lists
                n = kwargs['mean_period']
                if 'mean_period' in kwargs.keys():
                    n = kwargs['mean_period']
                    time_slots = list(chain.from_iterable([mean(time_slots[i:i+n])]*n for i in range(0,len(time_slots),n)))
                    concurrent_demands = list(chain.from_iterable([mean(concurrent_demands[i:i+n])]*n for i in range(0,len(concurrent_demands),n)))
                else:
                    # not smoothing results by taking mean
                    pass

                plot_dict[analyser.subject_class_name][json.dumps(link_type)][analyser.load_frac][link]['x_values'] = time_slots
                plot_dict[analyser.subject_class_name][json.dumps(link_type)][analyser.load_frac][link]['y_values'] = concurrent_demands

        load_to_meas_time = {} # collect measurement times for verical line plotting
        for analyser in analysers:
            self._check_analyser_valid(analyser)
            load_to_meas_time[analyser.load_frac] = [int(analyser.measurement_start_time/analyser.env.slot_size), int(analyser.measurement_end_time/analyser.env.slot_size)]


        figs = []
        for subject_class in plot_dict.keys():
            for link_type in plot_dict[subject_class].keys():
                for load in plot_dict[subject_class][link_type].keys():
                    fig = plot_dists.plot_val_line(plot_dict=plot_dict[subject_class][link_type][load], xlabel='Time Slot', ylabel='{} Load {} Concurrent Demands'.format(subject_class, str(round(load,2))), ylim=[0,None], linewidth=0.4, alpha=1, vertical_lines=load_to_meas_time[load], show_fig=False)
                    figs.append(fig)

        return figs


    def plot_demand_completion_time_vs_size_for_different_loads(self, *analysers):
        nested_dict = lambda: defaultdict(nested_dict)
        plot_dict = nested_dict()
        for analyser in analysers:
            self._check_analyser_valid(analyser)
            flow_completion_times = []
            flow_sizes = []
            for flow in analyser.completed_flow_dicts:
                flow_completion_times.append(flow['time_completed'] - flow['time_arrived'])
                flow_sizes.append(flow['size'])
            plot_dict[analyser.load_frac][analyser.subject_class_name]['x_values'] = flow_sizes
            plot_dict[analyser.load_frac][analyser.subject_class_name]['y_values'] = flow_completion_times

            # try:
                # plot_dict[analyser.load_frac][analyser.subject_class_name]['x_values'] = None
                # plot_dict[analyser.load_frac][analyser.subject_class_name]['y_values'] = None
            # except KeyError:
                # # not yet added this load
                # try:
                    # plot_dict[analyser.load_frac][analyser.subject_class_name] = {}
                    # plot_dict[analyser.load_frac][analyser.subject_class_name]['x_values'] = None
                    # plot_dict[analyser.load_frac][analyser.subject_class_name]['y_values'] = None
                # except KeyError:
                    # # not yet added this class
                    # plot_dict[analyser.load_frac] = {}
                    # plot_dict[analyser.load_frac][analyser.subject_class_name] = {}
                    # plot_dict[analyser.load_frac][analyser.subject_class_name]['x_values'] = None
                    # plot_dict[analyser.load_frac][analyser.subject_class_name]['y_values'] = None

        load_to_meas_time = {} # collect measurement times for verical line plotting
        for analyser in analysers:
            self._check_analyser_valid(analyser)
            load_to_meas_time[analyser.load_frac] = [int(analyser.measurement_start_time/analyser.env.slot_size), int(analyser.measurement_end_time/analyser.env.slot_size)]

            # flow_completion_times = []
            # flow_sizes = []
            # for flow in analyser.completed_flow_dicts:
                # flow_completion_times.append(flow['time_completed'] - flow['time_arrived'])
                # flow_sizes.append(flow['size'])
            # plot_dict[analyser.load_frac][analyser.subject_class_name]['x_values'] = flow_sizes
            # plot_dict[analyser.load_frac][analyser.subject_class_name]['y_values'] = flow_completion_times

        figs = []
        for load in plot_dict.keys():
            fig = plot_dists.plot_val_scatter(plot_dict=plot_dict[load], xlabel='Flow Size', ylabel='Load {} Flow Completion Time'.format(str(round(load,2))), alpha=0.5, logscale=True, show_fig=False)
            figs.append(fig)

        return figs


    ############################### BASRPT #################################

    def plot_average_fct_vs_basrpt_v(self, *analysers):
        classes = self._group_analyser_classes(*analysers)
        plot_dict = {_class: {'x_values': [], 'y_values': [], 'rand_vars': []} for _class in classes}

        for analyser in analysers:
            self._check_analyser_valid(analyser)
            plot_dict[analyser.subject_class_name]['x_values'].append(analyser.env.scheduler.V)
            plot_dict[analyser.subject_class_name]['y_values'].append(analyser.average_fct)
            plot_dict[analyser.subject_class_name]['rand_vars'].append(analyser.average_fct)

        # scatter
        fig1 = plot_dists.plot_val_scatter(plot_dict=plot_dict, xlabel='V', ylabel='Average FCT', show_fig=False)

        # complementary cdf
        fig2 = plot_dists.plot_val_cdf(plot_dict=plot_dict, xlabel='Average FCT', ylabel='Complementary CDF', complementary_cdf=True, show_fig=False)

        return [fig1, fig2]
    
    def plot_99th_percentile_fct_vs_basrpt_v(self, *analysers):
        classes = self._group_analyser_classes(*analysers)
        plot_dict = {_class: {'x_values': [], 'y_values': [], 'rand_vars': []} for _class in classes}

        for analyser in analysers:
            self._check_analyser_valid(analyser)
            plot_dict[analyser.subject_class_name]['x_values'].append(analyser.env.scheduler.V)
            plot_dict[analyser.subject_class_name]['y_values'].append(analyser.nn_fct)
            plot_dict[analyser.subject_class_name]['rand_vars'].append(analyser.nn_fct)

        # scatter
        fig1 = plot_dists.plot_val_scatter(plot_dict=plot_dict, xlabel='V', ylabel='99th Percentile FCT', show_fig=False)

        # complementary cdf
        fig2 = plot_dists.plot_val_cdf(plot_dict=plot_dict, xlabel='99th Percentile FCT', ylabel='Complementary CDF', complementary_cdf=True, show_fig=False)

        return [fig1, fig2]


    def plot_throughput_vs_basrpt_v(self, analysers):
        classes = self._group_analyser_classes(*analysers)
        plot_dict = {_class: {'x_values': [], 'y_values': [], 'rand_vars': []} for _class in classes}

        for analyser in analysers:
            self._check_analyser_valid(analyser)
            plot_dict[analyser.subject_class_name]['x_values'].append(analyser.env.scheduler.V)
            plot_dict[analyser.subject_class_name]['y_values'].append(analyser.throughput_abs)
            plot_dict[analyser.subject_class_name]['rand_vars'].append(analyser.throughput_abs)

        # scatter
        fig1 = plot_dists.plot_val_scatter(plot_dict=plot_dict, xlabel='V', ylabel='Throughput Rate', show_fig=False)

        # complementary cdf
        fig2 = plot_dists.plot_val_cdf(plot_dict=plot_dict, xlabel='Throughput Rate', ylabel='Complementary CDF', complementary_cdf=True, show_fig=False)

        return [fig1, fig2]













