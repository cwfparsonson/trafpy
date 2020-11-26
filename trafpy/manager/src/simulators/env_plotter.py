import inspect
from trafpy.generator.src.dists import plot_dists
import matplotlib.pyplot as plt


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

        # init plot dict
        plot_dict = {}
        for analyser in analysers:
            self._check_analyser_valid(analyser)
            try:
                plot_dict[analyser.load_frac][analyser.subject_class_name]['x_values'] = None
                plot_dict[analyser.load_frac][analyser.subject_class_name]['y_values'] = None
            except KeyError:
                # not yet added this load
                try:
                    plot_dict[analyser.load_frac][analyser.subject_class_name] = {}
                    plot_dict[analyser.load_frac][analyser.subject_class_name]['x_values'] = None
                    plot_dict[analyser.load_frac][analyser.subject_class_name]['y_values'] = None
                except KeyError:
                    # not yet added this class
                    plot_dict[analyser.load_frac] = {}
                    plot_dict[analyser.load_frac][analyser.subject_class_name] = {}
                    plot_dict[analyser.load_frac][analyser.subject_class_name]['x_values'] = None
                    plot_dict[analyser.load_frac][analyser.subject_class_name]['y_values'] = None


        load_to_meas_time = {} # collect measurement times for verical line plotting
        # plot src-dst queue evolution for each load
        for analyser in analysers:
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
        plot_dict = {}
        plot_dict2 = {_class: {'x_values': [], 'y_values': [], 'rand_vars': []} for _class in classes}
        for analyser in analysers:
            self._check_analyser_valid(analyser)
            plot_dict2[analyser.subject_class_name]['x_values'].append(analyser.load_frac)
            plot_dict2[analyser.subject_class_name]['y_values'].append(analyser.throughput_abs)
            plot_dict2[analyser.subject_class_name]['rand_vars'].append(analyser.throughput_abs)
            try:
                plot_dict[analyser.load_frac]['x_values'].append(analyser.subject_class_name)
                plot_dict[analyser.load_frac]['y_values'].append(analyser.throughput_abs)
            except KeyError:
                # not yet added this load
                try:
                    plot_dict[analyser.load_frac] = {}
                    plot_dict[analyser.load_frac]['x_values'].append(analyser.subject_class_name)
                    plot_dict[analyser.load_frac]['y_values'].append(analyser.throughput_abs)
                except KeyError:
                    # not yet added x and y values
                    plot_dict[analyser.load_frac] = {}
                    plot_dict[analyser.load_frac]['x_values'] = []
                    plot_dict[analyser.load_frac]['y_values'] = []
                    plot_dict[analyser.load_frac]['x_values'].append(analyser.subject_class_name)
                    plot_dict[analyser.load_frac]['y_values'].append(analyser.throughput_abs)

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

    def plot_num_concurrent_demands_vs_time_for_different_loads(self, *analysers):
        # init plot dict
        plot_dict = {}
        for analyser in analysers:
            if not analyser.env.track_grid_slot_evolution:
                raise Exception('Must set track_grid_slot_evolution=True when instantiating env simulation in order to plot number of concurrent demands.')
            self._check_analyser_valid(analyser)
            try:
                plot_dict[analyser.load_frac][analyser.subject_class_name]['x_values'] = None
                plot_dict[analyser.load_frac][analyser.subject_class_name]['y_values'] = None
            except KeyError:
                # not yet added this load
                try:
                    plot_dict[analyser.load_frac][analyser.subject_class_name] = {}
                    plot_dict[analyser.load_frac][analyser.subject_class_name]['x_values'] = None
                    plot_dict[analyser.load_frac][analyser.subject_class_name]['y_values'] = None
                except KeyError:
                    # not yet added this class
                    plot_dict[analyser.load_frac] = {}
                    plot_dict[analyser.load_frac][analyser.subject_class_name] = {}
                    plot_dict[analyser.load_frac][analyser.subject_class_name]['x_values'] = None
                    plot_dict[analyser.load_frac][analyser.subject_class_name]['y_values'] = None

        load_to_meas_time = {} # collect measurement times for verical line plotting
        for analyser in analysers:
            self._check_analyser_valid(analyser)
            load_to_meas_time[analyser.load_frac] = [int(analyser.measurement_start_time/analyser.env.slot_size), int(analyser.measurement_end_time/analyser.env.slot_size)]

            concurrent_demands = []
            time_slots = [t for t in range(analyser.grid_demands.shape[1])]
            for time_slot in time_slots:
                num_flows_in_network = 0
                for ep in analyser.env.grid_slot_dict.keys():
                    for channel in analyser.env.grid_slot_dict[ep].keys():
                        if analyser.env.grid_slot_dict[ep][channel]['demands'][time_slot] is not None:
                            # flow present in this flow slot
                            num_flows_in_network += 1
                        else:
                            # no flow present at this flow slot
                            pass
                concurrent_demands.append(num_flows_in_network)
            plot_dict[analyser.load_frac][analyser.subject_class_name]['x_values'] = time_slots
            plot_dict[analyser.load_frac][analyser.subject_class_name]['y_values'] = concurrent_demands

        figs = []
        for load in plot_dict.keys():
            fig = plot_dists.plot_val_line(plot_dict=plot_dict[load], xlabel='Time Slot', ylabel='Load {} Concurrent Demands'.format(str(round(load,2))), vertical_lines=load_to_meas_time[load], show_fig=False)
            figs.append(fig)

        return figs


    def plot_demand_completion_time_vs_size_for_different_loads(self, *analysers):
        # init plot dict
        plot_dict = {}
        for analyser in analysers:
            self._check_analyser_valid(analyser)
            try:
                plot_dict[analyser.load_frac][analyser.subject_class_name]['x_values'] = None
                plot_dict[analyser.load_frac][analyser.subject_class_name]['y_values'] = None
            except KeyError:
                # not yet added this load
                try:
                    plot_dict[analyser.load_frac][analyser.subject_class_name] = {}
                    plot_dict[analyser.load_frac][analyser.subject_class_name]['x_values'] = None
                    plot_dict[analyser.load_frac][analyser.subject_class_name]['y_values'] = None
                except KeyError:
                    # not yet added this class
                    plot_dict[analyser.load_frac] = {}
                    plot_dict[analyser.load_frac][analyser.subject_class_name] = {}
                    plot_dict[analyser.load_frac][analyser.subject_class_name]['x_values'] = None
                    plot_dict[analyser.load_frac][analyser.subject_class_name]['y_values'] = None

        load_to_meas_time = {} # collect measurement times for verical line plotting
        for analyser in analysers:
            self._check_analyser_valid(analyser)
            load_to_meas_time[analyser.load_frac] = [int(analyser.measurement_start_time/analyser.env.slot_size), int(analyser.measurement_end_time/analyser.env.slot_size)]

            flow_completion_times = []
            flow_sizes = []
            for flow in analyser.completed_flow_dicts:
                flow_completion_times.append(flow['time_completed'] - flow['time_arrived'])
                flow_sizes.append(flow['size'])
            plot_dict[analyser.load_frac][analyser.subject_class_name]['x_values'] = flow_sizes
            plot_dict[analyser.load_frac][analyser.subject_class_name]['y_values'] = flow_completion_times

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













