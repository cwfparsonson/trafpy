import inspect
from sqlitedict import SqliteDict
from trafpy.generator.src.dists import plot_dists
import matplotlib.pyplot as plt
from collections import defaultdict # use for initialising arbitrary length nested dict
import copy
import json
from statistics import mean
import math
from itertools import chain
from tabulate import tabulate
import pandas as pd
import numpy as np
import sigfig
from IPython.display import display
from sqlitedict import SqliteDict


class EnvPlotter:
    def __init__(self):
        pass


    def _check_analyser_valid(self, analyser):
        if inspect.isclass(analyser):
            raise Exception('Must instantiate EnvAnalyser class before passing to EnvPlotter.')

        if not analyser.computed_metrics:
            raise Exception('Must compute metrics with EnvAnalyser.compute_metrics() before passing to EnvPlotter.')


def get_summary_dict(analysers, headers, time_units='', info_units=''):
    summary_dict = {header: [] for header in headers}
    for analyser in analysers:
        # self._check_analyser_valid(analyser)
        summary_dict['Load'].append(round(analyser.load_frac, 2))
        summary_dict['Subject'].append(analyser.subject_class_name)
        summary_dict['T-Score'].append(analyser.t_score)
        summary_dict['Mean FCT ({})'.format(time_units)].append(round(analyser.mean_fct, 1))
        summary_dict['p99 FCT ({})'.format(time_units)].append(round(analyser.nn_fct, 1))
        summary_dict['Max FCT ({})'.format(time_units)].append(round(analyser.max_fct, 1))
        summary_dict['Throughput Frac'].append(sigfig.round(float(analyser.throughput_frac), sigfigs=6))
        summary_dict['Frac Flows Dropped'].append(sigfig.round(float(analyser.dropped_flow_frac), sigfigs=3))
        summary_dict['Frac Info Dropped'].append(sigfig.round(float(analyser.dropped_info_frac), sigfigs=3))
    return summary_dict

class EnvsPlotter:
    def __init__(self, time_units='', info_units=''):
        self.time_units = time_units
        self.info_units = info_units

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

    def display_t_score_table(self, *analysers, **kwargs):
        if 'display_table' not in kwargs:
            kwargs['display_table'] = True
        if 'plot_radar' not in kwargs:
            kwargs['plot_radar'] = True
        if 'figsize' not in kwargs:
            kwargs['figsize'] = (12.8, 9.6)
        if 'fill_alpha' not in kwargs:
            kwargs['fill_alpha'] = 0.05

        headers = ['Load', 
                   'Subject', 
                   'T-Score', 
                   'Mean FCT ({})'.format(self.time_units), 
                   'p99 FCT ({})'.format(self.time_units), 
                   'Max FCT ({})'.format(self.time_units), 
                   'Throughput Frac', 
                   'Frac Flows Dropped', 
                   'Frac Info Dropped']
        _summary_dict = get_summary_dict(analysers, headers, time_units=self.time_units, info_units=self.info_units)

        # sort by order of load
        index = np.argsort(_summary_dict['Load'])
        summary_dict = {}
        for key in _summary_dict.keys():
            summary_dict[key] = np.asarray(_summary_dict[key])[index]
        # print(summary_dict)

        # sort loads by t-score (best first)
        sorted_summary_dict = {header: [] for header in headers}
        # create winner table
        winner_table = {header: [] for header in headers if header != 'T-Score' and header != 'Subject'}
        winner_table['Load'] = list(np.unique(list(summary_dict['Load'])))

        num_loads = len(np.unique(list(summary_dict['Load'])))
        num_subjects = len(np.unique(list(summary_dict['Subject'])))
        i = 0 # for indexing each row in summary_dict
        i2 = 0 # for indexing the first instance of a new load row in summary_dict
        nested_dict = lambda: defaultdict(nested_dict)
        plot_dicts = [nested_dict() for _ in range(num_loads)]
        # determine if higher is better for each header
        is_higher_better = {}
        for header in headers:
            if header == 'T-Score' or header == 'Throughput Frac':
                is_higher_better[header] = True
            else:
                is_higher_better[header] = False
        # for load_idx in range(num_loads+1):
        for load_idx in range(num_loads):
            # sort t-scores for this load in descending order

            # find num_subjects for each load
            num_subjects = 0
            loads = []
            while summary_dict['Load'][i] in loads or len(loads) == 0:
                loads.append(summary_dict['Load'][i])
                num_subjects += 1
                i += 1
                if i == len(summary_dict['Load']):
                    # reached end of table
                    break
            # print('\nloads: {} | num subjects: {}'.format(loads, num_subjects))

            load_t_scores = summary_dict['T-Score'][i2:i2+num_subjects]
            load_t_score_indices = list(reversed(np.argsort(load_t_scores)))
            load_t_score_indices = [i2+_i for _i in load_t_score_indices]
            # print('t scores: {} | t score indices: {}'.format(load_t_scores, load_t_score_indices))

            # index by this order
            for header in headers:
                # update sorted summary dict
                sorted_summary_dict[header].extend(summary_dict[header][load_t_score_indices])
                
                # print('sorted summary dict:\n{}'.format(sorted_summary_dict))

                # update radar plot & winner table
                if header != 'Subject' and header != 'Load' and header != 'T-Score' and load_idx < num_loads:
                    # get classes and corresponding rand var values for this rand var
                    classes = summary_dict['Subject'][load_t_score_indices]
                    classes_rand_vars = summary_dict[header][load_t_score_indices]
                    # print('classes: {} | classes rand vars: {}'.format(classes, classes_rand_vars))

                    if is_higher_better[header]:
                        # max val wins
                        winner_val = max(classes_rand_vars)
                    else:
                        # min val wins
                        winner_val = min(classes_rand_vars)
                    winner_indices = sorted([u for u, v in enumerate(classes_rand_vars) if v == winner_val])
                    winner = classes[winner_indices[0]]
                    if len(winner_indices) > 1:
                        for winner_idx in winner_indices[1:]:
                            winner = winner + '+' + classes[winner_idx]
                    winner_table[header].append(winner)

                    # get min max range
                    min_val, max_val = min(classes_rand_vars), max(classes_rand_vars)
                    diff = max(max_val - min_val, 1e-9)
                    min_val -= (0.1*diff)
                    max_val += (0.1*diff)
                    _range = [min_val, max_val]
                    if not is_higher_better[header]:
                        # want lower (better) values on outer radar edge -> flip range
                        _range = _range[::-1]

                    plot_dicts[load_idx][header]['range'] = _range
                    for idx, _class in enumerate(classes):
                        plot_dicts[load_idx][header]['classes'][_class] = classes_rand_vars[idx]

            i2 += num_subjects 
             
        summary_dataframe = pd.DataFrame(sorted_summary_dict)
        winner_dataframe = pd.DataFrame(winner_table)
        if kwargs['display_table']:
            display(summary_dataframe)
            display(winner_dataframe)

        if kwargs['plot_radar']:
            loads = iter(np.unique(list(summary_dict['Load'])))
            for plot_dict in plot_dicts:
                # if len(list(plot_dict.keys())) != 0:
                radar = plot_dists.plot_radar(plot_dict, 
                                              title='Load {}'.format(next(loads)), 
                                              figsize=kwargs['figsize'],
                                              fill_alpha=kwargs['fill_alpha'],
                                              show_fig=True)

        return summary_dataframe, winner_dataframe


    def plot_t_score_scatter(self, *analysers, **kwargs):
        '''Plots performance indicators for T-scores.

        TODO:
            Currently plots FCT component vs. throughput component, but should add
            colour bar so can include dropped flow component (similar to final
            scatter in JLT plot).


        '''
        if 'gridlines' not in kwargs:
            kwargs['gridlines'] = True
        if 'aspect' not in kwargs:
            kwargs['aspect'] = 'auto'
        if 'figsize' not in kwargs:
            kwargs['figsize'] = (6.4, 4.8)
        if 'legend_ncol' not in kwargs:
            kwargs['legend_ncol'] = 1

        nested_dict = lambda: defaultdict(nested_dict)
        plot_dict = nested_dict()
        for analyser in analysers:
            self._check_analyser_valid(analyser)
            plot_dict[analyser.load_frac][analyser.subject_class_name]['x_values'] = analyser.throughput_component 
            plot_dict[analyser.load_frac][analyser.subject_class_name]['y_values'] = analyser.fct_component 


        figs = []
        for load in plot_dict.keys():
            fig = plot_dists.plot_val_scatter(plot_dict=plot_dict[load], xlabel='Load {} Throughput Component'.format(str(round(load,2))), ylabel='Load {} FCT Component'.format(str(round(load,2))), alpha=1.0, marker_size=60, logscale=True, gridlines=kwargs['gridlines'], aspect=kwargs['aspect'], figsize=kwargs['figsize'], legend_ncol=kwargs['legend_ncol'], show_fig=True)
            figs.append(fig)

        return figs

        

    def plot_mean_fct_vs_load(self, *analysers, **kwargs):
        '''
        *analysers (*args): Analyser objects whose metrics you wish to plot.
        '''
        if 'gridlines' not in kwargs:
            kwargs['gridlines'] = True
        if 'aspect' not in kwargs:
            kwargs['aspect'] = 'auto'
        if 'cdf_figsize' not in kwargs:
            kwargs['cdf_figsize'] = (6.4, 4.8)
        if 'scatter_figsize' not in kwargs:
            kwargs['scatter_figsize'] = (6.4, 4.8)
        if 'legend_ncol' not in kwargs:
            kwargs['legend_ncol'] = 1
        if 'logscale' not in kwargs:
            kwargs['logscale'] = False

        classes = self._group_analyser_classes(*analysers)
        plot_dict = {_class: {'x_values': [], 'y_values': [], 'rand_vars': []} for _class in classes}

        for analyser in analysers:
            self._check_analyser_valid(analyser)
            plot_dict[analyser.subject_class_name]['x_values'].append(analyser.load_frac)
            plot_dict[analyser.subject_class_name]['y_values'].append(analyser.mean_fct)
            plot_dict[analyser.subject_class_name]['rand_vars'].append(analyser.mean_fct)

        # scatter
        fig1 = plot_dists.plot_val_scatter(plot_dict=plot_dict, 
                                           xlabel='Load', 
                                           ylabel='Mean FCT ({})'.format(self.time_units), 
                                           plot_line=True,
                                           aspect=kwargs['aspect'],
                                           gridlines=kwargs['gridlines'],
                                           figsize=kwargs['scatter_figsize'],
                                           ylogscale=kwargs['logscale'],
                                           legend_ncol=kwargs['legend_ncol'],
                                           show_fig=True)

        # complementary cdf
        fig2 = plot_dists.plot_val_cdf(plot_dict=plot_dict, 
                                       xlabel='Mean FCT ({})'.format(self.time_units), 
                                       ylabel='Complementary CDF', 
                                       complementary_cdf=True, 
                                       aspect=kwargs['aspect'],
                                       gridlines=kwargs['gridlines'],
                                       logscale=kwargs['logscale'],
                                       figsize=kwargs['cdf_figsize'],
                                       legend_ncol=kwargs['legend_ncol'],
                                       show_fig=True)

        # % change
        for key in plot_dict.keys():
            init_idx = np.argmin(plot_dict[key]['x_values'])
            init_var = plot_dict[key]['y_values'][init_idx]
            plot_dict[key]['y_values'] = [plot_dict[key]['y_values'][i]/init_var for i in range(len(plot_dict[key]['y_values']))]
        fig3 = plot_dists.plot_val_scatter(plot_dict=plot_dict, 
                                           xlabel='Load', 
                                           ylabel='\u0394F Mean FCT', 
                                           gridlines=kwargs['gridlines'], 
                                           aspect=kwargs['aspect'], 
                                           plot_line=True,
                                           figsize=kwargs['scatter_figsize'], 
                                           ylogscale=kwargs['logscale'],
                                           legend_ncol=kwargs['legend_ncol'], 
                                           show_fig=True)


        return [fig1, fig2, fig3]

    def plot_99th_percentile_fct_vs_load(self, *analysers, **kwargs):
        if 'gridlines' not in kwargs:
            kwargs['gridlines'] = True
        if 'aspect' not in kwargs:
            kwargs['aspect'] = 'auto'
        if 'cdf_figsize' not in kwargs:
            kwargs['cdf_figsize'] = (6.4, 4.8)
        if 'scatter_figsize' not in kwargs:
            kwargs['scatter_figsize'] = (6.4, 4.8)
        if 'legend_ncol' not in kwargs:
            kwargs['legend_ncol'] = 1
        if 'logscale' not in kwargs:
            kwargs['logscale'] = False

        classes = self._group_analyser_classes(*analysers)
        plot_dict = {_class: {'x_values': [], 'y_values': [], 'rand_vars': []} for _class in classes}

        for analyser in analysers:
            self._check_analyser_valid(analyser)
            plot_dict[analyser.subject_class_name]['x_values'].append(analyser.load_frac)
            plot_dict[analyser.subject_class_name]['y_values'].append(analyser.nn_fct)
            plot_dict[analyser.subject_class_name]['rand_vars'].append(analyser.nn_fct)

        # scatter
        fig1 = plot_dists.plot_val_scatter(plot_dict=plot_dict, 
                                           xlabel='Load', 
                                           ylabel='p99 FCT ({})'.format(self.time_units), 
                                           gridlines=kwargs['gridlines'], 
                                           aspect=kwargs['aspect'], 
                                           figsize=kwargs['scatter_figsize'], 
                                           ylogscale=kwargs['logscale'],
                                           plot_line=True,
                                           legend_ncol=kwargs['legend_ncol'], 
                                           show_fig=True)

        # complementary cdf
        fig2 = plot_dists.plot_val_cdf(plot_dict=plot_dict, 
                                       xlabel='p99 FCT ({})'.format(self.time_units), 
                                       ylabel='Complementary CDF', 
                                       gridlines=kwargs['gridlines'], 
                                       aspect=kwargs['aspect'], 
                                       figsize=kwargs['cdf_figsize'], 
                                       logscale=kwargs['logscale'],
                                       legend_ncol=kwargs['legend_ncol'], 
                                       complementary_cdf=True, 
                                       show_fig=True)

        # % change
        for key in plot_dict.keys():
            init_idx = np.argmin(plot_dict[key]['x_values'])
            init_var = plot_dict[key]['y_values'][init_idx]
            plot_dict[key]['y_values'] = [plot_dict[key]['y_values'][i]/init_var for i in range(len(plot_dict[key]['y_values']))]
        fig3 = plot_dists.plot_val_scatter(plot_dict=plot_dict, 
                                           xlabel='Load', 
                                           ylabel='\u0394F p99 FCT', 
                                           gridlines=kwargs['gridlines'], 
                                           aspect=kwargs['aspect'], 
                                           plot_line=True,
                                           figsize=kwargs['scatter_figsize'], 
                                           legend_ncol=kwargs['legend_ncol'], 
                                           show_fig=True)

        return [fig1, fig2, fig3]

    def plot_max_fct_vs_load(self, *analysers, **kwargs):
        if 'gridlines' not in kwargs:
            kwargs['gridlines'] = True
        if 'aspect' not in kwargs:
            kwargs['aspect'] = 'auto'
        if 'cdf_figsize' not in kwargs:
            kwargs['cdf_figsize'] = (6.4, 4.8)
        if 'scatter_figsize' not in kwargs:
            kwargs['scatter_figsize'] = (6.4, 4.8)
        if 'legend_ncol' not in kwargs:
            kwargs['legend_ncol'] = 1
        if 'logscale' not in kwargs:
            kwargs['logscale'] = False

        classes = self._group_analyser_classes(*analysers)
        plot_dict = {_class: {'x_values': [], 'y_values': [], 'rand_vars': []} for _class in classes}

        for analyser in analysers:
            self._check_analyser_valid(analyser)
            plot_dict[analyser.subject_class_name]['x_values'].append(analyser.load_frac)
            plot_dict[analyser.subject_class_name]['y_values'].append(analyser.max_fct)
            plot_dict[analyser.subject_class_name]['rand_vars'].append(analyser.max_fct)

        # scatter
        fig1 = plot_dists.plot_val_scatter(plot_dict=plot_dict, 
                                           xlabel='Load', 
                                           ylabel='Max FCT ({})'.format(self.time_units), 
                                           gridlines=kwargs['gridlines'], 
                                           aspect=kwargs['aspect'], 
                                           ylogscale=kwargs['logscale'],
                                           plot_line=True,
                                           figsize=kwargs['scatter_figsize'], 
                                           legend_ncol=kwargs['legend_ncol'], 
                                           show_fig=True)

        # complementary cdf
        fig2 = plot_dists.plot_val_cdf(plot_dict=plot_dict, 
                                       xlabel='Max FCT ({})'.format(self.time_units), 
                                       ylabel='Complementary CDF', 
                                       gridlines=kwargs['gridlines'], 
                                       aspect=kwargs['aspect'], 
                                       figsize=kwargs['cdf_figsize'], 
                                       logscale=kwargs['logscale'],
                                       legend_ncol=kwargs['legend_ncol'], 
                                       complementary_cdf=True, 
                                       show_fig=True)

        # % change
        for key in plot_dict.keys():
            init_idx = np.argmin(plot_dict[key]['x_values'])
            init_var = plot_dict[key]['y_values'][init_idx]
            plot_dict[key]['y_values'] = [plot_dict[key]['y_values'][i]/init_var for i in range(len(plot_dict[key]['y_values']))]
        fig3 = plot_dists.plot_val_scatter(plot_dict=plot_dict, 
                                           xlabel='Load', 
                                           ylabel='\u0394F Max FCT', 
                                           gridlines=kwargs['gridlines'], 
                                           aspect=kwargs['aspect'], 
                                           plot_line=True,
                                           figsize=kwargs['scatter_figsize'], 
                                           legend_ncol=kwargs['legend_ncol'], 
                                           show_fig=True)

        return [fig1, fig2]

    def plot_fcts_cdf_for_different_loads(self, *analysers, **kwargs):
        if 'gridlines' not in kwargs:
            kwargs['gridlines'] = True
        if 'aspect' not in kwargs:
            kwargs['aspect'] = 'auto'
        if 'figsize' not in kwargs:
            kwargs['figsize'] = (6.4, 4.8)
        if 'legend_ncol' not in kwargs:
            kwargs['legend_ncol'] = 1

        # plot cdf of all fcts for each test subject for different loads
        classes = self._group_analyser_classes(*analysers)
        # plot_dict = {_class: {'x_values': [], 'rand_vars': []} for _class in classes}

        nested_dict = lambda: defaultdict(nested_dict)
        plot_dict = nested_dict()
        for analyser in analysers:
            self._check_analyser_valid(analyser)

            flow_completion_times = []
            if type(analyser.completed_flow_dicts) is str:
                # load from database
                completed_flow_dicts = SqliteDict(analyser.completed_flow_dicts)
            else:
                completed_flow_dicts = analyser.completed_flow_dicts
            for flow in completed_flow_dicts.values():
                flow_completion_times.append(flow['time_completed'] - flow['time_arrived'])
            if type(analyser.completed_flow_dicts) is str:
                completed_flow_dicts.close()

            plot_dict[analyser.load_frac][analyser.subject_class_name]['rand_vars'] = flow_completion_times

        # complementary cdf
        figs = []
        for load in plot_dict.keys():
            fig = plot_dists.plot_val_cdf(plot_dict=plot_dict[load], 
                                          xlabel='Load {} FCTs ({})'.format(round(load,2), self.time_units), 
                                          ylabel='Complementary CDF', 
                                          gridlines=kwargs['gridlines'], 
                                          aspect=kwargs['aspect'], 
                                          logscale=True, 
                                          plot_points=False, 
                                          complementary_cdf=True, 
                                          figsize=kwargs['figsize'], 
                                          legend_ncol=kwargs['legend_ncol'], 
                                          show_fig=True)
            figs.append(fig)

        return figs



    def plot_fraction_of_arrived_flows_dropped_vs_load(self, *analysers, **kwargs):
        if 'cdf_figsize' not in kwargs:
            kwargs['cdf_figsize'] = (6.4, 4.8)
        if 'scatter_figsize' not in kwargs:
            kwargs['scatter_figsize'] = (6.4, 4.8)
        if 'gridlines' not in kwargs:
            kwargs['gridlines'] = True
        if 'aspect' not in kwargs:
            kwargs['aspect'] = 'auto'
        if 'legend_ncol' not in kwargs:
            kwargs['legend_ncol'] = 1
        if 'logscale' not in kwargs:
            kwargs['logscale'] = False

        classes = self._group_analyser_classes(*analysers)
        plot_dict = {_class: {'x_values': [], 'y_values': [], 'rand_vars': []} for _class in classes}

        for analyser in analysers:
            self._check_analyser_valid(analyser)
            plot_dict[analyser.subject_class_name]['x_values'].append(analyser.load_frac)
            plot_dict[analyser.subject_class_name]['y_values'].append(analyser.dropped_flow_frac)
            plot_dict[analyser.subject_class_name]['rand_vars'].append(analyser.dropped_flow_frac)

        # scatter 
        fig1 = plot_dists.plot_val_scatter(plot_dict=plot_dict, 
                                           xlabel='Load', 
                                           ylabel='Fraction Flows Dropped', 
                                           gridlines=kwargs['gridlines'], 
                                           aspect=kwargs['aspect'], 
                                           plot_line=True,
                                           ylogscale=kwargs['logscale'],
                                           figsize=kwargs['scatter_figsize'], 
                                           legend_ncol=kwargs['legend_ncol'], 
                                           show_fig=True)

        # complementary cdf
        fig2 = plot_dists.plot_val_cdf(plot_dict=plot_dict, 
                                       xlabel='Fraction Flows Dropped', 
                                       ylabel='Complementary CDF', 
                                       complementary_cdf=True, 
                                       gridlines=kwargs['gridlines'], 
                                       logscale=kwargs['logscale'],
                                       aspect=kwargs['aspect'], 
                                       figsize=kwargs['cdf_figsize'], 
                                       legend_ncol=kwargs['legend_ncol'], 
                                       show_fig=True)

        # % change
        for key in plot_dict.keys():
            init_idx = np.argmin(plot_dict[key]['x_values'])
            init_var = plot_dict[key]['y_values'][init_idx]
            plot_dict[key]['y_values'] = [plot_dict[key]['y_values'][i]/init_var for i in range(len(plot_dict[key]['y_values']))]
        fig3 = plot_dists.plot_val_scatter(plot_dict=plot_dict, 
                                           xlabel='Load', 
                                           ylabel='\u0394F Flows Dropped', 
                                           gridlines=kwargs['gridlines'], 
                                           aspect=kwargs['aspect'], 
                                           plot_line=True,
                                           ylogscale=kwargs['logscale'],
                                           figsize=kwargs['scatter_figsize'], 
                                           legend_ncol=kwargs['legend_ncol'], 
                                           show_fig=True)

        return [fig1, fig2, fig3]

    def plot_fraction_of_arrived_info_dropped_vs_load(self, *analysers, **kwargs):
        if 'cdf_figsize' not in kwargs:
            kwargs['cdf_figsize'] = (6.4, 4.8)
        if 'scatter_figsize' not in kwargs:
            kwargs['scatter_figsize'] = (6.4, 4.8)
        if 'gridlines' not in kwargs:
            kwargs['gridlines'] = True
        if 'aspect' not in kwargs:
            kwargs['aspect'] = 'auto'
        if 'legend_ncol' not in kwargs:
            kwargs['legend_ncol'] = 1
        if 'logscale' not in kwargs:
            kwargs['logscale'] = False

        classes = self._group_analyser_classes(*analysers)
        plot_dict = {_class: {'x_values': [], 'y_values': [], 'rand_vars': []} for _class in classes}

        for analyser in analysers:
            self._check_analyser_valid(analyser)
            plot_dict[analyser.subject_class_name]['x_values'].append(analyser.load_frac)
            plot_dict[analyser.subject_class_name]['y_values'].append(analyser.dropped_info_frac)
            plot_dict[analyser.subject_class_name]['rand_vars'].append(analyser.dropped_info_frac)

        # scatter 
        fig1 = plot_dists.plot_val_scatter(plot_dict=plot_dict, 
                                           xlabel='Load', 
                                           ylabel='Fraction Info Dropped', 
                                           gridlines=kwargs['gridlines'], 
                                           aspect=kwargs['aspect'], 
                                           plot_line=True,
                                           ylogscale=kwargs['logscale'],
                                           figsize=kwargs['scatter_figsize'], 
                                           legend_ncol=kwargs['legend_ncol'], 
                                           show_fig=True)

        # complementary cdf
        fig2 = plot_dists.plot_val_cdf(plot_dict=plot_dict, 
                                       xlabel='Fraction Info Dropped', 
                                       ylabel='Complementary CDF', 
                                       gridlines=kwargs['gridlines'], 
                                       aspect=kwargs['aspect'], 
                                       figsize=kwargs['cdf_figsize'], 
                                       logscale=kwargs['logscale'],
                                       legend_ncol=kwargs['legend_ncol'], 
                                       complementary_cdf=True, 
                                       show_fig=True)

        # % change
        for key in plot_dict.keys():
            init_idx = np.argmin(plot_dict[key]['x_values'])
            init_var = plot_dict[key]['y_values'][init_idx]
            plot_dict[key]['y_values'] = [plot_dict[key]['y_values'][i]/init_var for i in range(len(plot_dict[key]['y_values']))]
        fig3 = plot_dists.plot_val_scatter(plot_dict=plot_dict, 
                                           xlabel='Load', 
                                           ylabel='\u0394F Info Dropped', 
                                           gridlines=kwargs['gridlines'], 
                                           aspect=kwargs['aspect'], 
                                           ylogscale=kwargs['logscale'],
                                           plot_line=True,
                                           figsize=kwargs['scatter_figsize'], 
                                           legend_ncol=kwargs['legend_ncol'], 
                                           show_fig=True)

        return [fig1, fig2, fig3]


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
            figs.append(plot_dists.plot_val_line(plot_dict=plot_dict[load], 
                                                 xlabel='Time', 
                                                 ylabel='Load {} {}-{} Queue Length'.format(str(round(load,2)), src, dst), 
                                                 ylim=load_to_ylim[load], 
                                                 vertical_lines=[load_to_meas_time[load][0], 
                                                 load_to_meas_time[load][1]], 
                                                 show_fig=True))
        
        return figs

    def plot_throughput_frac_vs_load(self, *analysers, **kwargs):
        if 'plot_bar_charts' not in kwargs:
            kwargs['plot_bar_charts'] = True
        if 'gridlines' not in kwargs:
            kwargs['gridlines'] = True
        if 'aspect' not in kwargs:
            kwargs['aspect'] = 'auto'
        if 'figsize' not in kwargs:
            kwargs['figsize'] = (6.4, 4.8)
        if 'legend_ncol' not in kwargs:
            kwargs['legend_ncol'] = 1

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
            plot_dict[analyser.load_frac]['y_values'].append(analyser.throughput_frac)

            plot_dict2[analyser.subject_class_name]['x_values'].append(analyser.load_frac)
            plot_dict2[analyser.subject_class_name]['y_values'].append(analyser.throughput_frac)
            plot_dict2[analyser.subject_class_name]['rand_vars'].append(analyser.throughput_frac)

        # individual bar chars
        figs = []
        if kwargs['plot_bar_charts']:
            for load in plot_dict.keys():
                figs.append(plot_dists.plot_val_bar(x_values=plot_dict[load]['x_values'], y_values=plot_dict[load]['y_values'], xlabel='Test Subject Class', ylabel='Load {} Throughput Rate'.format(str(round(load,2)), gridlines=kwargs['gridlines'], aspect=kwargs['aspect'], figsize=kwargs['figsize'], show_fig=True)))

        # scatter
        fig1 = plot_dists.plot_val_scatter(plot_dict=plot_dict2, 
                                           xlabel='Load', 
                                           ylabel='Throughput Fraction',
                                           gridlines=kwargs['gridlines'], 
                                           aspect=kwargs['aspect'], 
                                           plot_line=True,
                                           figsize=kwargs['figsize'], 
                                           legend_ncol=kwargs['legend_ncol'], 
                                           show_fig=True)

        # complementary cdf
        fig2 = plot_dists.plot_val_cdf(plot_dict=plot_dict2, 
                xlabel='Throughput Fraction',
                                       ylabel='Complementary CDF', 
                                       complementary_cdf=True, 
                                       gridlines=kwargs['gridlines'], 
                                       aspect=kwargs['aspect'], 
                                       figsize=kwargs['figsize'], 
                                       legend_ncol=kwargs['legend_ncol'], 
                                       show_fig=True)

        return [figs, fig1, fig2]


    def plot_throughput_rate_vs_load(self, *analysers, **kwargs):
        if 'plot_bar_charts' not in kwargs:
            kwargs['plot_bar_charts'] = True
        if 'gridlines' not in kwargs:
            kwargs['gridlines'] = True
        if 'aspect' not in kwargs:
            kwargs['aspect'] = 'auto'
        if 'figsize' not in kwargs:
            kwargs['figsize'] = (6.4, 4.8)
        if 'legend_ncol' not in kwargs:
            kwargs['legend_ncol'] = 1

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
        if kwargs['plot_bar_charts']:
            for load in plot_dict.keys():
                figs.append(plot_dists.plot_val_bar(x_values=plot_dict[load]['x_values'], y_values=plot_dict[load]['y_values'], xlabel='Test Subject Class', ylabel='Load {} Throughput Rate'.format(str(round(load,2)), gridlines=kwargs['gridlines'], aspect=kwargs['aspect'], figsize=kwargs['figsize'], show_fig=True)))

        # scatter
        fig1 = plot_dists.plot_val_scatter(plot_dict=plot_dict2, 
                                           xlabel='Load', 
                                           ylabel='Throughput Rate ({}/{})'.format(self.info_units, self.time_units), 
                                           gridlines=kwargs['gridlines'], 
                                           aspect=kwargs['aspect'], 
                                           plot_line=True,
                                           figsize=kwargs['figsize'], 
                                           legend_ncol=kwargs['legend_ncol'], 
                                           show_fig=True)

        # complementary cdf
        fig2 = plot_dists.plot_val_cdf(plot_dict=plot_dict2, 
                                       xlabel='Throughput Rate ({}/{})'.format(self.info_units, self.time_units), 
                                       ylabel='Complementary CDF', 
                                       complementary_cdf=True, 
                                       gridlines=kwargs['gridlines'], 
                                       aspect=kwargs['aspect'], 
                                       figsize=kwargs['figsize'], 
                                       legend_ncol=kwargs['legend_ncol'], 
                                       show_fig=True)

        
        # % change
        plot_dict3 = copy.deepcopy(plot_dict2)
        for key in plot_dict3.keys():
            init_idx = np.argmin(plot_dict3[key]['x_values'])
            init_var = plot_dict3[key]['y_values'][init_idx]
            plot_dict3[key]['y_values'] = [plot_dict3[key]['y_values'][i]/init_var for i in range(len(plot_dict3[key]['y_values']))]
        fig3 = plot_dists.plot_val_scatter(plot_dict=plot_dict3, 
                                           xlabel='Load', 
                                           ylabel='\u0394F Throughput Rate', 
                                           gridlines=kwargs['gridlines'], 
                                           aspect=kwargs['aspect'], 
                                           plot_line=True,
                                           figsize=kwargs['figsize'], 
                                           legend_ncol=kwargs['legend_ncol'], 
                                           show_fig=True)



        return [figs, fig1, fig2, fig3]


    # def plot_radars_for_different_loads(self, *analysers, **kwargs):
        # if 'figsize' not in kwargs:
            # kwargs['figsize'] = (6.4, 4.8)
        # if 'n_ordinate_levels' not in kwargs:
            # kwargs['n_ordinate_levels'] = 5
        # if 'fill' not in kwargs:
            # kwargs['fill'] = True
        # if 'legend_ncol' not in kwargs:
            # kwargs['legend_ncol'] = 1

        # classes = self._group_analyser_classes(*analysers)







    def plot_demand_slot_colour_grid_for_different_schedulers(self, *analysers, **kwargs):
        if 'gridlines' not in kwargs:
            kwargs['gridlines'] = True
        if 'aspect' not in kwargs:
            kwargs['aspect'] = 'auto'
        if 'figsize' not in kwargs:
            kwargs['figsize'] = (6.4, 4.8)
        if 'legend_ncol' not in kwargs:
            kwargs['legend_ncol'] = 1

        figs = []
        for analyser in analysers:
            if not analyser.env.track_grid_slot_evolution:
                raise Exception('Must set track_grid_slot_evolution=True when instantiating env simulation in order to plot grid.')
            self._check_analyser_valid(analyser)
            figs.append(plot_dists.plot_demand_slot_colour_grid(analyser.grid_demands, 
                                                                title=analyser.env.sim_name, 
                                                                xlim=None, 
                                                                show_fig=True, 
                                                                aspect=kwargs['aspect'], 
                                                                figsize=kwargs['figsize'], 
                                                                legend_ncol=kwargs['legend_ncol']))

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
        if 'alpha' not in kwargs:
            kwargs['alpha'] = 1
        if 'plot_legend' not in kwargs:
            kwargs['plot_legend'] = True
        if 'gridlines' not in kwargs:
            kwargs['gridlines'] = True
        if 'aspect' not in kwargs:
            kwargs['aspect'] = 'auto'
        if 'figsize' not in kwargs:
            kwargs['figsize'] = (6.4, 4.8)
        if 'legend_ncol' not in kwargs:
            kwargs['legend_ncol'] = 1

        nested_dict = lambda: defaultdict(nested_dict)
        plot_dict = nested_dict()

        link_types = []
        for analyser in analysers:
            if not analyser.env.track_link_utilisation_evolution:
                raise Exception('Must set track_link_utilisation_evolution=True when instantiating env simulation in order to plot link utilisation.')
            self._check_analyser_valid(analyser)

            if type(analyser.env.link_utilisation_dict) is str:
                # link_utilisation_dict is in database
                link_utilisation_dict = SqliteDict(analyser.env.link_utilisation_dict)
            else:
                link_utilisation_dict = analyser.env.link_utilisation_dict

            for link in link_utilisation_dict.keys():
                # assume node labels have format of edge type followed by underscore and int e.g. 'server_1', 'core_3' etc
                node1, node2 = json.loads(link)[0], json.loads(link)[1]
                idx1, idx2 = self.find_index_of_int_in_str(node1), self.find_index_of_int_in_str(node2)
                link_type = [node1[:idx1-1],node2[:idx2-1]]

                # OLD: Use to not consider different ports/directions on each link, but now should consider different directions separately...
                flip_link=False
                if link_type not in link_types:
                    flip_link=True
                    if link_type[::-1] not in link_types:
                        flip_link=False
                        link_types.append(link_type)
                if flip_link: 
                    link_type = link_type[::-1]

                time_slots = link_utilisation_dict[link]['time_slots']
                util = link_utilisation_dict[link]['util']


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
                # print('Plotting subject {} link type {}...'.format(subject_class, link_type))
                for load in plot_dict[subject_class][link_type].keys():
                    fig = plot_dists.plot_val_line(plot_dict=plot_dict[subject_class][link_type][load], 
                                                   xlabel='Time Slot', 
                                                   ylabel='{} Load {} Link Util'.format(subject_class, str(round(load,2))), 
                                                   ylim=[0,1], 
                                                   linewidth=0.4, 
                                                   title='{}-{}'.format(json.loads(link_type)[0], json.loads(link_type)[1]),
                                                   alpha=kwargs['alpha'], 
                                                   vertical_lines=load_to_meas_time[load], 
                                                   gridlines=kwargs['gridlines'], 
                                                   aspect=kwargs['aspect'], 
                                                   figsize=kwargs['figsize'], 
                                                   legend_ncol=kwargs['legend_ncol'], 
                                                   plot_legend=kwargs['plot_legend'], 
                                                   show_fig=True)
                    figs.append(fig)

        return figs


    def plot_link_concurrent_demands_vs_time_for_different_loads(self, *analysers, **kwargs):
        if 'gridlines' not in kwargs:
            kwargs['gridlines'] = True
        if 'aspect' not in kwargs:
            kwargs['aspect'] = 'auto'
        if 'figsize' not in kwargs:
            kwargs['figsize'] = (6.4, 4.8)
        if 'legend_ncol' not in kwargs:
            kwargs['legend_ncol'] = 1

        nested_dict = lambda: defaultdict(nested_dict)
        plot_dict = nested_dict()

        link_types = []
        for analyser in analysers:
            if not analyser.env.track_link_concurrent_demands_evolution:
                raise Exception('Must set track_link_concurrent_demands_evolution=True when instantiating env simulation in order to plot number of concurrent demands.')
            self._check_analyser_valid(analyser)

            if type(analyser.env.link_concurrent_demands_dict) is str:
                # link_utilisation_dict is in database
                link_concurrent_demands_dict = SqliteDict(analyser.env.link_concurrent_demands_dict)
            else:
                link_concurrent_demands_dict = analyser.env.link_concurrent_demands_dict
            for link in link_concurrent_demands_dict.keys():
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

                time_slots = link_concurrent_demands_dict[link]['time_slots']
                concurrent_demands = link_concurrent_demands_dict[link]['concurrent_demands']

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
                    fig = plot_dists.plot_val_line(plot_dict=plot_dict[subject_class][link_type][load], 
                                                   xlabel='Time Slot', 
                                                   ylabel='{} Load {} Concurrent Demands'.format(subject_class, str(round(load,2))), 
                                                   ylim=[0,None], 
                                                   linewidth=0.4, 
                                                   alpha=1, 
                                                   title='{}-{}'.format(json.loads(link_type)[0], json.loads(link_type)[1]),
                                                   vertical_lines=load_to_meas_time[load], 
                                                   gridlines=kwargs['gridlines'], 
                                                   aspect=kwargs['aspect'], 
                                                   figsize=kwargs['figsize'], 
                                                   legend_ncol=kwargs['legend_ncol'], 
                                                   show_fig=True)
                    figs.append(fig)

        return figs


    def plot_demand_completion_time_vs_size_for_different_loads(self, *analysers, **kwargs):
        if 'alpha' not in kwargs:
            kwargs['alpha'] = 1
        if 'gridlines' not in kwargs:
            kwargs['gridlines'] = True
        if 'aspect' not in kwargs:
            kwargs['aspect'] = 'auto'
        if 'figsize' not in kwargs:
            kwargs['figsize'] = (6.4, 4.8)
        if 'legend_ncol' not in kwargs:
            kwargs['legend_ncol'] = 1

        nested_dict = lambda: defaultdict(nested_dict)
        plot_dict = nested_dict()
        for analyser in analysers:
            self._check_analyser_valid(analyser)
            flow_completion_times = []
            flow_sizes = []
            if type(analyser.completed_flow_dicts) is str:
                # load from database
                completed_flow_dicts = SqliteDict(analyser.completed_flow_dicts)
            else:
                completed_flow_dicts = analyser.completed_flow_dicts
            for flow in completed_flow_dicts.values():
                flow_completion_times.append(flow['time_completed'] - flow['time_arrived'])
                flow_sizes.append(flow['size'])
            if type(analyser.completed_flow_dicts) is str:
                completed_flow_dicts.close()
            plot_dict[analyser.load_frac][analyser.subject_class_name]['x_values'] = flow_sizes
            plot_dict[analyser.load_frac][analyser.subject_class_name]['y_values'] = flow_completion_times

        load_to_meas_time = {} # collect measurement times for verical line plotting
        for analyser in analysers:
            self._check_analyser_valid(analyser)
            load_to_meas_time[analyser.load_frac] = [int(analyser.measurement_start_time/analyser.env.slot_size), int(analyser.measurement_end_time/analyser.env.slot_size)]

        figs = []
        for load in plot_dict.keys():
            fig = plot_dists.plot_val_scatter(plot_dict=plot_dict[load], 
                                              xlabel='Flow Size ({})'.format(self.info_units), 
                                              ylabel='Load {} FCT ({})'.format(str(round(load,2)), self.time_units), 
                                              alpha=kwargs['alpha'], 
                                              logscale=True, 
                                              gridlines=kwargs['gridlines'], 
                                              aspect=kwargs['aspect'], 
                                              figsize=kwargs['figsize'], 
                                              legend_ncol=kwargs['legend_ncol'], 
                                              show_fig=True)
            figs.append(fig)

        return figs








    # ############################### BASRPT #################################

    # def plot_mean_fct_vs_basrpt_v(self, *analysers):
        # classes = self._group_analyser_classes(*analysers)
        # plot_dict = {_class: {'x_values': [], 'y_values': [], 'rand_vars': []} for _class in classes}

        # for analyser in analysers:
            # self._check_analyser_valid(analyser)
            # plot_dict[analyser.subject_class_name]['x_values'].append(analyser.env.scheduler.V)
            # plot_dict[analyser.subject_class_name]['y_values'].append(analyser.mean_fct)
            # plot_dict[analyser.subject_class_name]['rand_vars'].append(analyser.mean_fct)

        # # scatter
        # fig1 = plot_dists.plot_val_scatter(plot_dict=plot_dict, xlabel='V', ylabel='Average FCT', show_fig=True)

        # # complementary cdf
        # fig2 = plot_dists.plot_val_cdf(plot_dict=plot_dict, xlabel='Average FCT', ylabel='Complementary CDF', complementary_cdf=True, show_fig=True)

        # return [fig1, fig2]
    
    # def plot_99th_percentile_fct_vs_basrpt_v(self, *analysers):
        # classes = self._group_analyser_classes(*analysers)
        # plot_dict = {_class: {'x_values': [], 'y_values': [], 'rand_vars': []} for _class in classes}

        # for analyser in analysers:
            # self._check_analyser_valid(analyser)
            # plot_dict[analyser.subject_class_name]['x_values'].append(analyser.env.scheduler.V)
            # plot_dict[analyser.subject_class_name]['y_values'].append(analyser.nn_fct)
            # plot_dict[analyser.subject_class_name]['rand_vars'].append(analyser.nn_fct)

        # # scatter
        # fig1 = plot_dists.plot_val_scatter(plot_dict=plot_dict, xlabel='V', ylabel='99th Percentile FCT', show_fig=True)

        # # complementary cdf
        # fig2 = plot_dists.plot_val_cdf(plot_dict=plot_dict, xlabel='99th Percentile FCT', ylabel='Complementary CDF', complementary_cdf=True, show_fig=True)

        # return [fig1, fig2]


    # def plot_throughput_vs_basrpt_v(self, analysers):
        # classes = self._group_analyser_classes(*analysers)
        # plot_dict = {_class: {'x_values': [], 'y_values': [], 'rand_vars': []} for _class in classes}

        # for analyser in analysers:
            # self._check_analyser_valid(analyser)
            # plot_dict[analyser.subject_class_name]['x_values'].append(analyser.env.scheduler.V)
            # plot_dict[analyser.subject_class_name]['y_values'].append(analyser.throughput_abs)
            # plot_dict[analyser.subject_class_name]['rand_vars'].append(analyser.throughput_abs)

        # # scatter
        # fig1 = plot_dists.plot_val_scatter(plot_dict=plot_dict, xlabel='V', ylabel='Throughput Rate', show_fig=True)

        # # complementary cdf
        # fig2 = plot_dists.plot_val_cdf(plot_dict=plot_dict, xlabel='Throughput Rate', ylabel='Complementary CDF', complementary_cdf=True, show_fig=True)

        # return [fig1, fig2]













