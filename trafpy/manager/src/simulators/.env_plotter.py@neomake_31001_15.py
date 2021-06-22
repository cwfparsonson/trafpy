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
import os


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
        summary_dict['Throughput'].append(sigfig.round(float(analyser.throughput_frac), sigfigs=6))
        summary_dict['Throughput ({}/{})'.format(info_units, time_units)].append(sigfig.round(float(analyser.throughput_abs), sigfigs=6))
        summary_dict['Frac Flows Accepted'].append(sigfig.round(float(1-analyser.dropped_flow_frac), sigfigs=6))
        summary_dict['Frac Info Accepted'].append(sigfig.round(float(1-analyser.dropped_info_frac), sigfigs=6))

        if analyser.env.job_centric:
            summary_dict['Mean JCT ({})'.format(time_units)].append(round(analyser.mean_jct, 1))
            summary_dict['p99 JCT ({})'.format(time_units)].append(round(analyser.nn_jct, 1))
            summary_dict['Max JCT ({})'.format(time_units)].append(round(analyser.max_jct, 1))
            summary_dict['Frac Jobs Accepted'].append(sigfig.round(float(1-analyser.dropped_job_frac), sigfigs=6))

    return summary_dict

class EnvsPlotter:
    def __init__(self, time_units='', info_units='', path_to_save=None):
        self.time_units = time_units
        self.info_units = info_units

        if path_to_save is not None:
            folder_name = path_to_save.split('/')[-1] # want unique to make copying to local machine easier across multiple envs_plotter folders
            path_to_save = path_to_save + '/envs_plotter_{}'.format(folder_name)
            if os.path.exists(path_to_save):
                pass
            else:
                os.mkdir(path_to_save)
        self.path_to_save = path_to_save


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

    def construct_tables(self, *analysers, **kwargs):
        '''
        Constructs summary dataframes and tables and (optionally) prints out
        and displays these tables. Optionally also visualises these summary
        data with radar plots.

        '''
        if 'display_tables' not in kwargs:
            kwargs['display_tables'] = True
        if 'plot_radar' not in kwargs:
            kwargs['plot_radar'] = True
        if 'figsize' not in kwargs:
            kwargs['figsize'] = (12.8, 9.6)
        if 'font_size' not in kwargs:
            kwargs['font_size'] = 10
        if 'fill_alpha' not in kwargs:
            kwargs['fill_alpha'] = 0.05
        if 'print_latex_tables' not in kwargs:
            kwargs['print_latex_tables'] = True # path to write excel tables to

        headers = ['Load', 
                   'Subject', 
                   'T-Score', 
                   'Mean FCT ({})'.format(self.time_units), 
                   'p99 FCT ({})'.format(self.time_units), 
                   'Max FCT ({})'.format(self.time_units), 
                   'Throughput', 
                   'Throughput ({}/{})'.format(self.info_units, self.time_units),
                   'Frac Flows Accepted', 
                   'Frac Info Accepted']
        for analyser in analysers:
            job_centric = analyser.env.job_centric
            break
        if job_centric:
            headers.append('Mean JCT ({})'.format(self.time_units))
            headers.append('p99 JCT ({})'.format(self.time_units))
            headers.append('Max JCT ({})'.format(self.time_units))
            headers.append('Frac Jobs Accepted')


        _summary_dict = get_summary_dict(analysers, headers, time_units=self.time_units, info_units=self.info_units)

        # sort by order of load
        index = np.argsort(_summary_dict['Load'])
        summary_dict = {}
        for key in _summary_dict.keys():
            summary_dict[key] = np.asarray(_summary_dict[key])[index]
        # print(summary_dict)

        # sort loads by subject name (alphabetical)
        sorted_summary_dict = {header: [] for header in headers}
        # create winner table
        winner_table = {header: [] for header in headers if header != 'T-Score' and header != 'Subject' and 'Throughput (' not in header}
        winner_table['Load'] = list(np.unique(list(summary_dict['Load'])))
        # create ranking table
        ranking_table = {header: [] for header in headers if header != 'T-Score' and header != 'Subject' and 'Throughput (' not in header}
        ranking_table['Load'] = list(np.unique(list(summary_dict['Load'])))
        # create excel tables
        # excel_summary_table = {header: {subject: [] for subject in np.unique(list(summary_dict['Subject']))} for header in headers if header != 'T-Score' and header != 'Subject'}
        # excel_summary_table['Load'] = list(np.unique(list(summary_dict['Load'])))
        # excel_ranking_table = {header: {subject: [] for subject in np.unique(list(summary_dict['Subject']))} for header in headers if header != 'T-Score' and header != 'Subject'}
        # excel_ranking_table['Load'] = list(np.unique(list(summary_dict['Load'])))
        latex_winner_table = {header: [] for header in headers if header != 'T-Score' and header != 'Subject' and 'Throughput (' not in header}
        latex_winner_table['Load'] = list(np.unique(list(summary_dict['Load'])))

        num_loads = len(np.unique(list(summary_dict['Load'])))
        num_subjects = len(np.unique(list(summary_dict['Subject'])))
        i = 0 # for indexing each row in summary_dict
        i2 = 0 # for indexing the first instance of a new load row in summary_dict
        nested_dict = lambda: defaultdict(nested_dict)
        plot_dicts = [nested_dict() for _ in range(num_loads)]
        # determine if higher is better for each header
        is_higher_better = {}
        for header in headers:
            if header == 'T-Score' or 'Throughput' in header or 'Accepted' in header:
                is_higher_better[header] = True
            else:
                is_higher_better[header] = False
        # for load_idx in range(num_loads+1):
        for load_idx in range(num_loads):
            # sort subjects for this load 

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

            # load_t_scores = summary_dict['T-Score'][i2:i2+num_subjects]
            # load_t_score_indices = list(reversed(np.argsort(load_t_scores)))
            # load_t_score_indices = [i2+_i for _i in load_t_score_indices]
            # print('t scores: {} | t score indices: {}'.format(load_t_scores, load_t_score_indices))
            subjects = summary_dict['Subject'][i2:i2+num_subjects]
            load_subject_indices = list(np.argsort(subjects))
            load_subject_indices = [i2+_i for _i in load_subject_indices]

            # index by this order
            for header in headers:
                # update sorted summary dict
                sorted_summary_dict[header].extend(summary_dict[header][load_subject_indices])
                
                # print('sorted summary dict:\n{}'.format(sorted_summary_dict))

                # update radar plot & winner table
                if header != 'Subject' and header != 'Load' and header != 'T-Score' and 'Throughput (' not in header and load_idx < num_loads:
                    # get classes and corresponding rand var values for this rand var
                    classes = summary_dict['Subject'][load_subject_indices]
                    classes_rand_vars = summary_dict[header][load_subject_indices]
                    # print('classes: {} | classes rand vars: {}'.format(classes, classes_rand_vars))

                    # winner table
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


                    # ranking table
                    if is_higher_better[header]:
                        # max values better
                        ranking_indices = np.argsort(classes_rand_vars)[::-1]
                    else:
                        # low values better
                        ranking_indices = np.argsort(classes_rand_vars)
                    ranking_vals = np.asarray(classes_rand_vars)[ranking_indices]
                    ranking_classes = np.asarray(classes)[ranking_indices]
                    # find loser -> make this baseline
                    baseline_val, baseline_class = ranking_vals[-1], ranking_classes[-1]
                    # find performance improvement of other classes relative to baseline
                    relative_performance = []
                    for val in ranking_vals[:-1]:
                        diff = val - baseline_val
                        if diff != 0:
                            perf = sigfig.round(float((diff/baseline_val) * 100), sigfigs=4)
                        else:
                            # same performance as baseline val
                            perf = 0
                        relative_performance.append(perf)
                    relative_performance_iterator = iter(relative_performance)
                    ranks = ''
                    for _class in ranking_classes[:-1]:
                        rank = _class + '({}%) | '.format(sigfig.round(next(relative_performance_iterator), sigfigs=4))
                        ranks += rank
                    ranks += baseline_class
                    ranking_table[header].append(ranks)

                    # add winner w/ relative performance to excel winner table
                    # winner = ranking_classes[0]
                    perf = relative_performance[0]
                    latex_winner_table[header].append('{}, {}%'.format(winner, perf))

                    # # sort ranking classes by alphabet and add to excel ranking table
                    # excel_indices = np.argsort(ranking_classes[:-1])
                    # relative_performance_iterator = iter(np.asarray(relative_performance)[excel_indices])
                    # for _class in np.asarray(ranking_classes[:-1])[excel_indices]:
                        # excel_ranking_table[header][_class].append(next(relative_performance_iterator))
                    # excel_ranking_table[header][baseline_class].append(0)
                    

                    # excel summary table
                    # excel_indices = np.argsort(classes)
                    # for _class, val in zip(np.asarray(classes)[excel_indices], np.asarray(classes_rand_vars)[excel_indices]):
                        # excel_summary_table[header][_class].append(val)
                        







                    # radar plot
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

        # change headers of winner table to appropriate headers
        keys = list(latex_winner_table.keys())
        for key in keys:
            new_key = None
            if 'Mean' in key:
                new_key = 'Mean FCT'
            elif 'p99' in key:
                new_key = 'p99 FCT'
            elif 'Max' in key:
                new_key = 'Max FCT'
            elif 'Throughput Frac' in key:
                new_key = 'Throughput'
            elif 'Throughput (' in key:
                new_key = 'Throughput Rate'
            elif 'Flows Accepted' in key:
                new_key = 'Flows Accepted'
            elif 'Info Accepted' in key:
                new_key = 'Info Accepted'
            else:
                # don't need to adjust
                pass
            if new_key is not None and new_key != 'Throughput Rate':
                latex_winner_table[new_key] = latex_winner_table.pop(key)

        # change headers of latex summary table
        latex_summary_table = copy.deepcopy(sorted_summary_dict)
        keys = list(latex_summary_table.keys())
        for key in keys:
            new_key = None
            if 'Mean' in key:
                new_key = 'Mean FCT ($\mu$s)'
            elif 'p99' in key:
                new_key = 'p99 FCT ($\mu$s)'
            elif 'Max' in key:
                new_key = 'Max FCT ($\mu$s)'
            elif 'Throughput' in key:
                new_key = 'Throughput (Frac)'
            elif 'Throughput (' in key:
                new_key = 'Throughput Rate (B/$\mu$s)'
            elif 'Flows Accepted' in key:
                new_key = 'Flows Accepted (Frac)'
            elif 'Info Accepted' in key:
                new_key = 'Info Accepted (Frac)'
            else:
                # don't need to adjust
                pass
            if new_key is not None:
                latex_summary_table[new_key] = latex_summary_table.pop(key)
        del latex_summary_table['T-Score']
            
             
        summary_dataframe = pd.DataFrame(sorted_summary_dict)
        winner_dataframe = pd.DataFrame(winner_table)
        ranking_dataframe = pd.DataFrame(ranking_table)
        latex_summary_dataframe = pd.DataFrame(latex_summary_table)
        latex_winner_dataframe = pd.DataFrame(latex_winner_table)


        if kwargs['print_latex_tables']:
            print('\n\nSummary latex table:')
            print(latex_summary_dataframe.to_latex(index=False, multicolumn=True, escape=False))
            print('\n\nWinner latex table:')
            print(latex_winner_dataframe.to_latex(index=False, multicolumn=True))

        if self.path_to_save is not None:
            summary_dataframe.to_pickle(self.path_to_save+'/summary_dataframe.pkl')
            latex_summary_dataframe.to_latex(index=False, multicolumn=True, buf=self.path_to_save+'/latex_summary_table', escape=False)
            latex_winner_dataframe.to_latex(index=False, multicolumn=True, buf=self.path_to_save+'/latex_winner_table', escape=False)
            print('Saved tables and dataframes to {}'.format(self.path_to_save))



        if kwargs['display_tables']:
            display(summary_dataframe)
            display(winner_dataframe)
            display(ranking_dataframe)
            # display(excel_summary_dataframe)
            # display(excel_ranking_dataframe)
            # display(latex_winner_dataframe)

        if kwargs['plot_radar']:
            loads = iter(np.unique(list(summary_dict['Load'])))
            for plot_dict in plot_dicts:
                # if len(list(plot_dict.keys())) != 0:
                load = next(loads)
                radar = plot_dists.plot_radar(plot_dict, 
                                              title='Load {}'.format(load), 
                                              figsize=kwargs['figsize'],
                                              font_size=kwargs['font_size'],
                                              fill_alpha=kwargs['fill_alpha'],
                                              show_fig=True)
                if self.path_to_save is not None:
                    radar.savefig(self.path_to_save+'/radar_load_{}.png'.format(round(load, 2)), bbox_inches='tight')


        return summary_dataframe, winner_dataframe, ranking_dataframe


    def plot_t_score_scatter(self, *analysers, **kwargs):
        '''Plots performance indicators for T-scores.

        '''
        if 'gridlines' not in kwargs:
            kwargs['gridlines'] = True
        if 'aspect' not in kwargs:
            kwargs['aspect'] = 'auto'
        if 'figsize' not in kwargs:
            kwargs['figsize'] = (6.4, 4.8)
        if 'font_size' not in kwargs:
            kwargs['font_size'] = 10
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
            fig = plot_dists.plot_val_scatter(plot_dict=plot_dict[load], xlabel='Load {} Throughput Component'.format(str(round(load,2))), ylabel='Load {} FCT Component'.format(str(round(load,2))), alpha=1.0, marker_size=60, logscale=True, gridlines=kwargs['gridlines'], aspect=kwargs['aspect'], font_size=kwargs['font_size'], figsize=kwargs['figsize'], legend_ncol=kwargs['legend_ncol'], show_fig=True)
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
        if 'font_size' not in kwargs:
            kwargs['font_size'] = 10
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
                                           font_size=kwargs['font_size'],
                                           ylogscale=kwargs['logscale'],
                                           legend_ncol=kwargs['legend_ncol'],
                                           show_fig=True)

        # complementary cdf
        fig2 = plot_dists.plot_val_cdf(plot_dict=plot_dict, 
                                       xlabel='Mean FCT ({})'.format(self.time_units), 
                                       ylabel='C-CDF', 
                                       complementary_cdf=True, 
                                       aspect=kwargs['aspect'],
                                       gridlines=kwargs['gridlines'],
                                       logscale=kwargs['logscale'],
                                       figsize=kwargs['cdf_figsize'],
                                       font_size=kwargs['font_size'],
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
                                           font_size=kwargs['font_size'],
                                           ylogscale=kwargs['logscale'],
                                           legend_ncol=kwargs['legend_ncol'], 
                                           show_fig=True)

        if self.path_to_save is not None:
            fig1.savefig(self.path_to_save+'/mean_fct_vs_load_scatter.png', bbox_inches='tight')
            fig2.savefig(self.path_to_save+'/mean_fct_vs_load_complementary_cdf.png', bbox_inches='tight')
            fig3.savefig(self.path_to_save+'/mean_fct_vs_load_change.png', bbox_inches='tight')


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
        if 'font_size' not in kwargs:
            kwargs['font_size'] = 10
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
                                           font_size=kwargs['font_size'],
                                           ylogscale=kwargs['logscale'],
                                           plot_line=True,
                                           legend_ncol=kwargs['legend_ncol'], 
                                           show_fig=True)

        # complementary cdf
        fig2 = plot_dists.plot_val_cdf(plot_dict=plot_dict, 
                                       xlabel='p99 FCT ({})'.format(self.time_units), 
                                       ylabel='C-CDF', 
                                       gridlines=kwargs['gridlines'], 
                                       aspect=kwargs['aspect'], 
                                       figsize=kwargs['cdf_figsize'], 
                                       font_size=kwargs['font_size'],
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
                                           font_size=kwargs['font_size'],
                                           legend_ncol=kwargs['legend_ncol'], 
                                           show_fig=True)

        if self.path_to_save is not None:
            fig1.savefig(self.path_to_save+'/p99_fct_vs_load_scatter.png', bbox_inches='tight')
            fig2.savefig(self.path_to_save+'/p99_fct_vs_load_complementary_cdf.png', bbox_inches='tight')
            fig3.savefig(self.path_to_save+'/p99_fct_vs_load_change.png', bbox_inches='tight')

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
        if 'font_size' not in kwargs:
            kwargs['font_size'] = 10
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
                                           font_size=kwargs['font_size'],
                                           legend_ncol=kwargs['legend_ncol'], 
                                           show_fig=True)

        # complementary cdf
        fig2 = plot_dists.plot_val_cdf(plot_dict=plot_dict, 
                                       xlabel='Max FCT ({})'.format(self.time_units), 
                                       ylabel='C-CDF', 
                                       gridlines=kwargs['gridlines'], 
                                       aspect=kwargs['aspect'], 
                                       figsize=kwargs['cdf_figsize'], 
                                       font_size=kwargs['font_size'],
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
                                           font_size=kwargs['font_size'],
                                           legend_ncol=kwargs['legend_ncol'], 
                                           show_fig=True)

        if self.path_to_save is not None:
            fig1.savefig(self.path_to_save+'/max_fct_vs_load_scatter.png', bbox_inches='tight')
            fig2.savefig(self.path_to_save+'/max_fct_vs_load_complementary_cdf.png', bbox_inches='tight')
            fig3.savefig(self.path_to_save+'/max_fct_vs_load_change.png', bbox_inches='tight')

        return [fig1, fig2]

    def plot_fcts_cdf_for_different_loads(self, *analysers, **kwargs):
        if 'gridlines' not in kwargs:
            kwargs['gridlines'] = True
        if 'aspect' not in kwargs:
            kwargs['aspect'] = 'auto'
        if 'figsize' not in kwargs:
            kwargs['figsize'] = (6.4, 4.8)
        if 'font_size' not in kwargs:
            kwargs['font_size'] = 10
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
                                          ylabel='C-CDF', 
                                          gridlines=kwargs['gridlines'], 
                                          aspect=kwargs['aspect'], 
                                          logscale=True, 
                                          plot_points=False, 
                                          complementary_cdf=True, 
                                          figsize=kwargs['figsize'], 
                                          font_size=kwargs['font_size'],
                                          legend_ncol=kwargs['legend_ncol'], 
                                          show_fig=True)
            if self.path_to_save is not None:
                fig.savefig(self.path_to_save+'/fcts_cdf_load_{}.png'.format(round(load, 2)), bbox_inches='tight')
            figs.append(fig)

        return figs



    def plot_fraction_of_arrived_flows_dropped_vs_load(self, *analysers, **kwargs):
        if 'cdf_figsize' not in kwargs:
            kwargs['cdf_figsize'] = (6.4, 4.8)
        if 'scatter_figsize' not in kwargs:
            kwargs['scatter_figsize'] = (6.4, 4.8)
        if 'font_size' not in kwargs:
            kwargs['font_size'] = 10
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
                                           ylabel='Flows Dropped', 
                                           gridlines=kwargs['gridlines'], 
                                           aspect=kwargs['aspect'], 
                                           plot_line=True,
                                           ylogscale=kwargs['logscale'],
                                           figsize=kwargs['scatter_figsize'], 
                                           font_size=kwargs['font_size'],
                                           legend_ncol=kwargs['legend_ncol'], 
                                           show_fig=True)

        # complementary cdf
        fig2 = plot_dists.plot_val_cdf(plot_dict=plot_dict, 
                                       xlabel='Flows Dropped', 
                                       ylabel='C-CDF', 
                                       complementary_cdf=True, 
                                       gridlines=kwargs['gridlines'], 
                                       logscale=kwargs['logscale'],
                                       aspect=kwargs['aspect'], 
                                       figsize=kwargs['cdf_figsize'], 
                                       font_size=kwargs['font_size'],
                                       legend_ncol=kwargs['legend_ncol'], 
                                       show_fig=True)

        # % change
        for key in plot_dict.keys():
            init_idx = np.argmin(plot_dict[key]['x_values'])
            init_var = plot_dict[key]['y_values'][init_idx]
            if init_var != 0: 
                plot_dict[key]['y_values'] = [plot_dict[key]['y_values'][i]/init_var for i in range(len(plot_dict[key]['y_values']))]
            else:
                # was 0 flows dropped and now not -> infinitely worse
                plot_dict[key]['y_values'] = [-float('inf') for _ in range(len(plot_dict[key]['y_values']))]
        fig3 = plot_dists.plot_val_scatter(plot_dict=plot_dict, 
                                           xlabel='Load', 
                                           ylabel='\u0394F Flows Dropped', 
                                           gridlines=kwargs['gridlines'], 
                                           aspect=kwargs['aspect'], 
                                           plot_line=True,
                                           ylogscale=kwargs['logscale'],
                                           figsize=kwargs['scatter_figsize'], 
                                           font_size=kwargs['font_size'],
                                           legend_ncol=kwargs['legend_ncol'], 
                                           show_fig=True)

        if self.path_to_save is not None:
            fig1.savefig(self.path_to_save+'/frac_arrived_flows_dropped_vs_load_scatter.png', bbox_inches='tight')
            fig2.savefig(self.path_to_save+'/frac_arrived_flows_dropped_vs_load_complementary_cdf.png', bbox_inches='tight')
            fig3.savefig(self.path_to_save+'/frac_arrived_flows_dropped_vs_load_change.png', bbox_inches='tight')

        return [fig1, fig2, fig3]

    def plot_fraction_of_arrived_info_dropped_vs_load(self, *analysers, **kwargs):
        if 'cdf_figsize' not in kwargs:
            kwargs['cdf_figsize'] = (6.4, 4.8)
        if 'scatter_figsize' not in kwargs:
            kwargs['scatter_figsize'] = (6.4, 4.8)
        if 'font_size' not in kwargs:
            kwargs['font_size'] = 10
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
                                           ylabel='Info Dropped', 
                                           gridlines=kwargs['gridlines'], 
                                           aspect=kwargs['aspect'], 
                                           plot_line=True,
                                           ylogscale=kwargs['logscale'],
                                           figsize=kwargs['scatter_figsize'], 
                                           font_size=kwargs['font_size'],
                                           legend_ncol=kwargs['legend_ncol'], 
                                           show_fig=True)

        # complementary cdf
        fig2 = plot_dists.plot_val_cdf(plot_dict=plot_dict, 
                                       xlabel='Info Dropped', 
                                       ylabel='C-CDF', 
                                       gridlines=kwargs['gridlines'], 
                                       aspect=kwargs['aspect'], 
                                       figsize=kwargs['cdf_figsize'], 
                                       font_size=kwargs['font_size'],
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
                                           font_size=kwargs['font_size'],
                                           legend_ncol=kwargs['legend_ncol'], 
                                           show_fig=True)

        if self.path_to_save is not None:
            fig1.savefig(self.path_to_save+'/frac_arrived_info_dropped_vs_load_scatter.png', bbox_inches='tight')
            fig2.savefig(self.path_to_save+'/frac_arrived_info_dropped_vs_load_complementary_cdf.png', bbox_inches='tight')
            fig3.savefig(self.path_to_save+'/frac_arrived_info_dropped_vs_load_change.png', bbox_inches='tight')

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
        if 'cdf_figsize' not in kwargs:
            kwargs['cdf_figsize'] = (6.4, 4.8)
        if 'scatter_figsize' not in kwargs:
            kwargs['scatter_figsize'] = (6.4, 4.8)
        if 'font_size' not in kwargs:
            kwargs['font_size'] = 10
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
                figs.append(plot_dists.plot_val_bar(x_values=plot_dict[load]['x_values'], y_values=plot_dict[load]['y_values'], xlabel='Test Subject Class', ylabel='Load {} Throughput ({}/{})'.format(round(load,2), self.info_units, self.time_units), gridlines=kwargs['gridlines'], aspect=kwargs['aspect'], figsize=kwargs['figsize'], show_fig=True))

        # scatter
        fig1 = plot_dists.plot_val_scatter(plot_dict=plot_dict2, 
                                           xlabel='Load', 
                                           ylabel='Throughput',
                                           gridlines=kwargs['gridlines'], 
                                           aspect=kwargs['aspect'], 
                                           plot_line=True,
                                           figsize=kwargs['scatter_figsize'], 
                                           font_size=kwargs['font_size'],
                                           legend_ncol=kwargs['legend_ncol'], 
                                           show_fig=True)

        # complementary cdf
        fig2 = plot_dists.plot_val_cdf(plot_dict=plot_dict2, 
                                       xlabel='Throughput',
                                       ylabel='C-CDF', 
                                       complementary_cdf=True, 
                                       gridlines=kwargs['gridlines'], 
                                       aspect=kwargs['aspect'], 
                                       figsize=kwargs['cdf_figsize'], 
                                       font_size=kwargs['font_size'],
                                       legend_ncol=kwargs['legend_ncol'], 
                                       show_fig=True)

        if self.path_to_save is not None:
            fig1.savefig(self.path_to_save+'/throughput_frac_vs_load_scatter.png', bbox_inches='tight')
            fig2.savefig(self.path_to_save+'/throughput_frac_vs_load_complementary_cdf.png', bbox_inches='tight')

        return [figs, fig1, fig2]


    def plot_throughput_rate_vs_load(self, *analysers, **kwargs):
        if 'plot_bar_charts' not in kwargs:
            kwargs['plot_bar_charts'] = True
        if 'gridlines' not in kwargs:
            kwargs['gridlines'] = True
        if 'aspect' not in kwargs:
            kwargs['aspect'] = 'auto'
        if 'cdf_figsize' not in kwargs:
            kwargs['cdf_figsize'] = (6.4, 4.8)
        if 'scatter_figsize' not in kwargs:
            kwargs['scatter_figsize'] = (6.4, 4.8)
        if 'font_size' not in kwargs:
            kwargs['font_size'] = 10
        if 'legend_ncol' not in kwargs:
            kwargs['legend_ncol'] = 1
        if 'use_scientific_notation' not in kwargs:
            kwargs['use_scientific_notation'] = False

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
                figs.append(plot_dists.plot_val_bar(x_values=plot_dict[load]['x_values'], y_values=plot_dict[load]['y_values'], xlabel='Test Subject Class', ylabel='Load {} Throughput ({}/{})'.format(round(load,2), self.info_units, self.time_units), gridlines=kwargs['gridlines'], aspect=kwargs['aspect'], figsize=kwargs['figsize'], show_fig=True))

        # scatter
        fig1 = plot_dists.plot_val_scatter(plot_dict=plot_dict2, 
                                           xlabel='Load', 
                                           ylabel='Throughput ({}/{})'.format(self.info_units, self.time_units), 
                                           gridlines=kwargs['gridlines'], 
                                           aspect=kwargs['aspect'], 
                                           plot_line=True,
                                           use_scientific_notation_yaxis=kwargs['use_scientific_notation'],
                                           figsize=kwargs['scatter_figsize'], 
                                           font_size=kwargs['font_size'],
                                           legend_ncol=kwargs['legend_ncol'], 
                                           show_fig=True)

        # complementary cdf
        fig2 = plot_dists.plot_val_cdf(plot_dict=plot_dict2, 
                                       xlabel='Throughput ({}/{})'.format(self.info_units, self.time_units), 
                                       ylabel='C-CDF', 
                                       complementary_cdf=True, 
                                       gridlines=kwargs['gridlines'], 
                                       use_scientific_notation=kwargs['use_scientific_notation'],
                                       aspect=kwargs['aspect'], 
                                       figsize=kwargs['cdf_figsize'], 
                                       font_size=kwargs['font_size'],
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
                                           ylabel='\u0394F Throughput', 
                                           gridlines=kwargs['gridlines'], 
                                           aspect=kwargs['aspect'], 
                                           plot_line=True,
                                           figsize=kwargs['scatter_figsize'], 
                                           font_size=kwargs['font_size'],
                                           legend_ncol=kwargs['legend_ncol'], 
                                           show_fig=True)

        if self.path_to_save is not None:
            fig1.savefig(self.path_to_save+'/throughput_rate_vs_load_scatter.png', bbox_inches='tight')
            fig2.savefig(self.path_to_save+'/throughput_rate_vs_load_complementary_cdf.png', bbox_inches='tight')
            fig3.savefig(self.path_to_save+'/throughput_rate_vs_load_change.png', bbox_inches='tight')



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
        if 'font_size' not in kwargs:
            kwargs['font_size'] = 10
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
                                                                font_size=kwargs['font_size'],
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
        if 'font_size' not in kwargs:
            kwargs['font_size'] = 10
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
                                                   font_size=kwargs['font_size'],
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
        if 'font_size' not in kwargs:
            kwargs['font_size'] = 10
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
                                                   font_size=kwargs['font_size'],
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
        if 'font_size' not in kwargs:
            kwargs['font_size'] = 10
        if 'legend_ncol' not in kwargs:
            kwargs['legend_ncol'] = 1
        if 'use_scientific_notation' not in kwargs:
            kwargs['use_scientific_notation'] = False

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
                                              use_scientific_notation_yaxis=kwargs['use_scientific_notation_yaxis'],
                                              figsize=kwargs['figsize'], 
                                              font_size=kwargs['font_size'],
                                              legend_ncol=kwargs['legend_ncol'], 
                                              show_fig=True)
            if self.path_to_save is not None:
                fig.savefig(self.path_to_save+'/demand_completion_time_vs_size_load_{}.png'.format(round(load, 2)), bbox_inches='tight')
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
        # fig2 = plot_dists.plot_val_cdf(plot_dict=plot_dict, xlabel='Average FCT', ylabel='C-CDF', complementary_cdf=True, show_fig=True)

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
        # fig2 = plot_dists.plot_val_cdf(plot_dict=plot_dict, xlabel='99th Percentile FCT', ylabel='C-CDF', complementary_cdf=True, show_fig=True)

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
        # fig1 = plot_dists.plot_val_scatter(plot_dict=plot_dict, xlabel='V', ylabel='Throughput', show_fig=True)

        # # complementary cdf
        # fig2 = plot_dists.plot_val_cdf(plot_dict=plot_dict, xlabel='Throughput', ylabel='C-CDF', complementary_cdf=True, show_fig=True)

        # return [fig1, fig2]


    def plot_jcts_cdf_for_different_loads(self, *analysers, **kwargs):
        if 'gridlines' not in kwargs:
            kwargs['gridlines'] = True
        if 'aspect' not in kwargs:
            kwargs['aspect'] = 'auto'
        if 'figsize' not in kwargs:
            kwargs['figsize'] = (6.4, 4.8)
        if 'font_size' not in kwargs:
            kwargs['font_size'] = 10
        if 'legend_ncol' not in kwargs:
            kwargs['legend_ncol'] = 1

        # plot cdf of all jcts for each test subject for different loads
        classes = self._group_analyser_classes(*analysers)
        # plot_dict = {_class: {'x_values': [], 'rand_vars': []} for _class in classes}

        nested_dict = lambda: defaultdict(nested_dict)
        plot_dict = nested_dict()
        for analyser in analysers:
            self._check_analyser_valid(analyser)

            job_completion_times = []
            if type(analyser.completed_job_dicts) is str:
                # load from database
                completed_job_dicts = SqliteDict(analyser.completed_job_dicts)
            else:
                completed_job_dicts = analyser.completed_job_dicts
            for job in completed_job_dicts.values():
                job_completion_times.append(job['time_completed'] - job['time_arrived'])
            if type(analyser.completed_job_dicts) is str:
                completed_job_dicts.close()

            plot_dict[analyser.load_frac][analyser.subject_class_name]['rand_vars'] = job_completion_times

        # complementary cdf
        figs = []
        for load in plot_dict.keys():
            fig = plot_dists.plot_val_cdf(plot_dict=plot_dict[load], 
                                          xlabel='Load {} JCTs ({})'.format(round(load,2), self.time_units), 
                                          ylabel='C-CDF', 
                                          gridlines=kwargs['gridlines'], 
                                          aspect=kwargs['aspect'], 
                                          logscale=True, 
                                          plot_points=False, 
                                          complementary_cdf=True, 
                                          figsize=kwargs['figsize'], 
                                          font_size=kwargs['font_size'],
                                          legend_ncol=kwargs['legend_ncol'], 
                                          show_fig=True)
            if self.path_to_save is not None:
                fig.savefig(self.path_to_save+'/jcts_cdf_load_{}.png'.format(round(load, 2)), bbox_inches='tight')
            figs.append(fig)

        return figs


    def plot_mean_jct_vs_load(self, *analysers, **kwargs):
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
        if 'font_size' not in kwargs:
            kwargs['font_size'] = 10
        if 'legend_ncol' not in kwargs:
            kwargs['legend_ncol'] = 1
        if 'logscale' not in kwargs:
            kwargs['logscale'] = False

        classes = self._group_analyser_classes(*analysers)
        plot_dict = {_class: {'x_values': [], 'y_values': [], 'rand_vars': []} for _class in classes}

        for analyser in analysers:
            self._check_analyser_valid(analyser)
            plot_dict[analyser.subject_class_name]['x_values'].append(analyser.load_frac)
            plot_dict[analyser.subject_class_name]['y_values'].append(analyser.mean_jct)
            plot_dict[analyser.subject_class_name]['rand_vars'].append(analyser.mean_jct)

        # scatter
        fig1 = plot_dists.plot_val_scatter(plot_dict=plot_dict, 
                                           xlabel='Load', 
                                           ylabel='Mean JCT ({})'.format(self.time_units), 
                                           plot_line=True,
                                           aspect=kwargs['aspect'],
                                           gridlines=kwargs['gridlines'],
                                           figsize=kwargs['scatter_figsize'],
                                           font_size=kwargs['font_size'],
                                           ylogscale=kwargs['logscale'],
                                           legend_ncol=kwargs['legend_ncol'],
                                           show_fig=True)

        # complementary cdf
        fig2 = plot_dists.plot_val_cdf(plot_dict=plot_dict, 
                                       xlabel='Mean JCT ({})'.format(self.time_units), 
                                       ylabel='C-CDF', 
                                       complementary_cdf=True, 
                                       aspect=kwargs['aspect'],
                                       gridlines=kwargs['gridlines'],
                                       logscale=kwargs['logscale'],
                                       figsize=kwargs['cdf_figsize'],
                                       font_size=kwargs['font_size'],
                                       legend_ncol=kwargs['legend_ncol'],
                                       show_fig=True)

        # % change
        for key in plot_dict.keys():
            init_idx = np.argmin(plot_dict[key]['x_values'])
            init_var = plot_dict[key]['y_values'][init_idx]
            plot_dict[key]['y_values'] = [plot_dict[key]['y_values'][i]/init_var for i in range(len(plot_dict[key]['y_values']))]
        fig3 = plot_dists.plot_val_scatter(plot_dict=plot_dict, 
                                           xlabel='Load', 
                                           ylabel='\u0394F Mean JCT', 
                                           gridlines=kwargs['gridlines'], 
                                           aspect=kwargs['aspect'], 
                                           plot_line=True,
                                           figsize=kwargs['scatter_figsize'], 
                                           font_size=kwargs['font_size'],
                                           ylogscale=kwargs['logscale'],
                                           legend_ncol=kwargs['legend_ncol'], 
                                           show_fig=True)

        if self.path_to_save is not None:
            fig1.savefig(self.path_to_save+'/mean_jct_vs_load_scatter.png', bbox_inches='tight')
            fig2.savefig(self.path_to_save+'/mean_jct_vs_load_complementary_cdf.png', bbox_inches='tight')
            fig3.savefig(self.path_to_save+'/mean_jct_vs_load_change.png', bbox_inches='tight')

        return [fig1, fig2, fig3]



    def plot_99th_percentile_jct_vs_load(self, *analysers, **kwargs):
        if 'gridlines' not in kwargs:
            kwargs['gridlines'] = True
        if 'aspect' not in kwargs:
            kwargs['aspect'] = 'auto'
        if 'cdf_figsize' not in kwargs:
            kwargs['cdf_figsize'] = (6.4, 4.8)
        if 'scatter_figsize' not in kwargs:
            kwargs['scatter_figsize'] = (6.4, 4.8)
        if 'font_size' not in kwargs:
            kwargs['font_size'] = 10
        if 'legend_ncol' not in kwargs:
            kwargs['legend_ncol'] = 1
        if 'logscale' not in kwargs:
            kwargs['logscale'] = False

        classes = self._group_analyser_classes(*analysers)
        plot_dict = {_class: {'x_values': [], 'y_values': [], 'rand_vars': []} for _class in classes}

        for analyser in analysers:
            self._check_analyser_valid(analyser)
            plot_dict[analyser.subject_class_name]['x_values'].append(analyser.load_frac)
            plot_dict[analyser.subject_class_name]['y_values'].append(analyser.nn_jct)
            plot_dict[analyser.subject_class_name]['rand_vars'].append(analyser.nn_jct)

        # scatter
        fig1 = plot_dists.plot_val_scatter(plot_dict=plot_dict, 
                                           xlabel='Load', 
                                           ylabel='p99 JCT ({})'.format(self.time_units), 
                                           gridlines=kwargs['gridlines'], 
                                           aspect=kwargs['aspect'], 
                                           figsize=kwargs['scatter_figsize'], 
                                           font_size=kwargs['font_size'],
                                           ylogscale=kwargs['logscale'],
                                           plot_line=True,
                                           legend_ncol=kwargs['legend_ncol'], 
                                           show_fig=True)

        # complementary cdf
        fig2 = plot_dists.plot_val_cdf(plot_dict=plot_dict, 
                                       xlabel='p99 JCT ({})'.format(self.time_units), 
                                       ylabel='C-CDF', 
                                       gridlines=kwargs['gridlines'], 
                                       aspect=kwargs['aspect'], 
                                       figsize=kwargs['cdf_figsize'], 
                                       font_size=kwargs['font_size'],
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
                                           ylabel='\u0394F p99 JCT', 
                                           gridlines=kwargs['gridlines'], 
                                           aspect=kwargs['aspect'], 
                                           plot_line=True,
                                           figsize=kwargs['scatter_figsize'], 
                                           font_size=kwargs['font_size'],
                                           legend_ncol=kwargs['legend_ncol'], 
                                           show_fig=True)

        if self.path_to_save is not None:
            fig1.savefig(self.path_to_save+'/p99_jct_vs_load_scatter.png', bbox_inches='tight')
            fig2.savefig(self.path_to_save+'/p99_jct_vs_load_complementary_cdf.png', bbox_inches='tight')
            fig3.savefig(self.path_to_save+'/p99_jct_vs_load_change.png', bbox_inches='tight')

        return [fig1, fig2, fig3]


    def plot_max_jct_vs_load(self, *analysers, **kwargs):
        if 'gridlines' not in kwargs:
            kwargs['gridlines'] = True
        if 'aspect' not in kwargs:
            kwargs['aspect'] = 'auto'
        if 'cdf_figsize' not in kwargs:
            kwargs['cdf_figsize'] = (6.4, 4.8)
        if 'scatter_figsize' not in kwargs:
            kwargs['scatter_figsize'] = (6.4, 4.8)
        if 'font_size' not in kwargs:
            kwargs['font_size'] = 10
        if 'legend_ncol' not in kwargs:
            kwargs['legend_ncol'] = 1
        if 'logscale' not in kwargs:
            kwargs['logscale'] = False

        classes = self._group_analyser_classes(*analysers)
        plot_dict = {_class: {'x_values': [], 'y_values': [], 'rand_vars': []} for _class in classes}

        for analyser in analysers:
            self._check_analyser_valid(analyser)
            plot_dict[analyser.subject_class_name]['x_values'].append(analyser.load_frac)
            plot_dict[analyser.subject_class_name]['y_values'].append(analyser.max_jct)
            plot_dict[analyser.subject_class_name]['rand_vars'].append(analyser.max_jct)

        # scatter
        fig1 = plot_dists.plot_val_scatter(plot_dict=plot_dict, 
                                           xlabel='Load', 
                                           ylabel='Max JCT ({})'.format(self.time_units), 
                                           gridlines=kwargs['gridlines'], 
                                           aspect=kwargs['aspect'], 
                                           ylogscale=kwargs['logscale'],
                                           plot_line=True,
                                           figsize=kwargs['scatter_figsize'], 
                                           font_size=kwargs['font_size'],
                                           legend_ncol=kwargs['legend_ncol'], 
                                           show_fig=True)

        # complementary cdf
        fig2 = plot_dists.plot_val_cdf(plot_dict=plot_dict, 
                                       xlabel='Max JCT ({})'.format(self.time_units), 
                                       ylabel='C-CDF', 
                                       gridlines=kwargs['gridlines'], 
                                       aspect=kwargs['aspect'], 
                                       figsize=kwargs['cdf_figsize'], 
                                       font_size=kwargs['font_size'],
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
                                           ylabel='\u0394F Max JCT', 
                                           gridlines=kwargs['gridlines'], 
                                           aspect=kwargs['aspect'], 
                                           plot_line=True,
                                           figsize=kwargs['scatter_figsize'], 
                                           font_size=kwargs['font_size'],
                                           legend_ncol=kwargs['legend_ncol'], 
                                           show_fig=True)

        if self.path_to_save is not None:
            fig1.savefig(self.path_to_save+'/max_jct_vs_load_scatter.png', bbox_inches='tight')
            fig2.savefig(self.path_to_save+'/max_jct_vs_load_complementary_cdf.png', bbox_inches='tight')
            fig3.savefig(self.path_to_save+'/max_jct_vs_load_change.png', bbox_inches='tight')

        return [fig1, fig2]


    def plot_fraction_of_arrived_jobs_dropped_vs_load(self, *analysers, **kwargs):
        if 'cdf_figsize' not in kwargs:
            kwargs['cdf_figsize'] = (6.4, 4.8)
        if 'scatter_figsize' not in kwargs:
            kwargs['scatter_figsize'] = (6.4, 4.8)
        if 'font_size' not in kwargs:
            kwargs['font_size'] = 10
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
            plot_dict[analyser.subject_class_name]['y_values'].append(analyser.dropped_job_frac)
            plot_dict[analyser.subject_class_name]['rand_vars'].append(analyser.dropped_job_frac)

        # scatter 
        fig1 = plot_dists.plot_val_scatter(plot_dict=plot_dict, 
                                           xlabel='Load', 
                                           ylabel='Jobs Dropped', 
                                           gridlines=kwargs['gridlines'], 
                                           aspect=kwargs['aspect'], 
                                           plot_line=True,
                                           ylogscale=kwargs['logscale'],
                                           figsize=kwargs['scatter_figsize'], 
                                           font_size=kwargs['font_size'],
                                           legend_ncol=kwargs['legend_ncol'], 
                                           show_fig=True)

        # complementary cdf
        fig2 = plot_dists.plot_val_cdf(plot_dict=plot_dict, 
                                       xlabel='Jobs Dropped', 
                                       ylabel='C-CDF', 
                                       complementary_cdf=True, 
                                       gridlines=kwargs['gridlines'], 
                                       logscale=kwargs['logscale'],
                                       aspect=kwargs['aspect'], 
                                       figsize=kwargs['cdf_figsize'], 
                                       font_size=kwargs['font_size'],
                                       legend_ncol=kwargs['legend_ncol'], 
                                       show_fig=True)

        # % change
        for key in plot_dict.keys():
            init_idx = np.argmin(plot_dict[key]['x_values'])
            init_var = plot_dict[key]['y_values'][init_idx]
            plot_dict[key]['y_values'] = [plot_dict[key]['y_values'][i]/init_var for i in range(len(plot_dict[key]['y_values']))]
        fig3 = plot_dists.plot_val_scatter(plot_dict=plot_dict, 
                                           xlabel='Load', 
                                           ylabel='\u0394F Jobs Dropped', 
                                           gridlines=kwargs['gridlines'], 
                                           aspect=kwargs['aspect'], 
                                           plot_line=True,
                                           ylogscale=kwargs['logscale'],
                                           figsize=kwargs['scatter_figsize'], 
                                           font_size=kwargs['font_size'],
                                           legend_ncol=kwargs['legend_ncol'], 
                                           show_fig=True)

        if self.path_to_save is not None:
            fig1.savefig(self.path_to_save+'/frac_arrived_jobs_dropped_vs_load_scatter.png', bbox_inches='tight')
            fig2.savefig(self.path_to_save+'/frac_arrived_jobs_dropped_vs_load_complementary_cdf.png', bbox_inches='tight')
            fig3.savefig(self.path_to_save+'/frac_arrived_jobs_dropped_vs_load_change.png', bbox_inches='tight')

        return [fig1, fig2, fig3]





























# FUNCS
def plot_summary_dict_params(summary_dicts, 
                             error_summary_dicts=None,
                              dependent_var_name='Random Variable', 
                              dependent_var_display_name=None,
                              control_var_name='Param', 
                              loads=[0.1, 0.5, 0.9], 
                              subjects_to_plot='all',
                              path_to_save=None,
                              **kwargs):
    '''
    This function is for when you have used `envs_plotter' to plot individual
    simulation statistics for various loads, and now wish to do some plots
    across multiple simulation statistics for cross-simulation analysis. Note
    that this notebook requires you having set a path_to_save string when
    calling `envs_plotter' so that an `envs_plotter'_ directory contains a
    summary_dataframe.pkl file.

    summary_dicts should be a dict of dicts. The keys of summary_dicts should
    be the parameter values of the control_var_name (e.g. for
    rack_dist_sensitivity benchmark, would be 0.2, 0.4, 0.6, and 0.8 for 20%,
    40%, 60%, and 80% intra-rack traffic).  The values are the originial
    summary_dataframe.pkl file which has been loaded as a pandas dataframe and
    converted to a dict (pf.read_pickle(path_to_summary_dataframe.pkl).to_dict)

    The keys of these individual summary dicts will be e.g. 'Load', 'Subject',
    'Mean FCT' etc., and the values will be the corresponding values.

    If error_summary_dicts is not None, should be a summary_dict formatted in same
    way as summary_dicts but containing the error (e.g. std) for each stat.
    '''
    if dependent_var_display_name is None:
        dependent_var_display_name = dependent_var_name
    if 'gridlines' not in kwargs:
        kwargs['gridlines'] = True
    if 'aspect' not in kwargs:
        kwargs['aspect'] = 'auto'
    if 'cdf_figsize' not in kwargs:
        kwargs['cdf_figsize'] = (6.4, 4.8)
    if 'scatter_figsize' not in kwargs:
        kwargs['scatter_figsize'] = (6.4, 4.8)
    if 'font_size' not in kwargs:
        kwargs['font_size'] = 10
    if 'legend_ncol' not in kwargs:
        kwargs['legend_ncol'] = 1
    if 'logscale' not in kwargs:
        kwargs['logscale'] = False
    if 'use_scientific_notation' not in kwargs:
        kwargs['use_scientific_notation'] = False
    if 'replace_summary_dict_dependent_var_name' not in kwargs: # useful to replace to e.g. replace Throughput Frac with Throughput, but less useful if want to keep e.g. Mean FCT units
        kwargs['replace_summary_dict_dependent_var_name'] = False

    all_subjects = np.unique(list(summary_dicts[list(summary_dicts.keys())[0]]['Subject'].values()))
    ghost_subjects = [] # track which (if any) subjects not plotted so retain consistent plot class colours
    if type(subjects_to_plot) is str:
        if subjects_to_plot == 'all':
            subjects_to_plot = all_subjects
             # no ghost subjects needed to keep same class plot colours
    else:
        # check if any subjects not included, add to ghost_subjects if not included
        for subject in all_subjects:
            if subject not in subjects_to_plot:
                ghost_subjects.append(subject)

    if control_var_name != 'Load':
        # check loads are in summary_dicts, since somtimes get e.g. 0.89 instead of 0.9
        _loads = []
        summ_loads =  list(summary_dicts[list(summary_dicts.keys())[0]]['Load'].values())
        for load in loads:
            if load not in summ_loads:
                # load not in summary_dicts
                # see if +- 0.01 helps
                load_plus, load_minus = load + 0.01, load - 0.01
                if load_plus in summ_loads:
                    _loads.append(load_plus)
                elif load_minus in summ_loads:
                    _loads.append(load_minus)
                else:
                    raise Exception('Unable to find load {} in summary dict loads {}'.format(load, summ_loads))
            else:
                # load in summary_dicts
                _loads.append(load)
        # update list of loads to ensure compatible with summary_dicts
        loads = _loads


    
    if control_var_name != 'Load':
        # plot separate plot for each load
        num_plots = len(loads)
    else:
        # only plot one plot for all loads
        num_plots = 1
    counter = 0
    while counter != num_plots:
        # new plot

        if control_var_name != 'Load':
            # separate plots for each load. Get load
            load = loads[counter]

        # get rand var key
        rand_var_key = None
        for key in summary_dicts[list(summary_dicts.keys())[0]].keys():
            if dependent_var_name != 'Throughput':
                if dependent_var_name in key:
                    rand_var_key = copy.deepcopy(key)
                    break
            else:
                # have Throughput and Throughput (xxx), so need to distinguish
                if key == dependent_var_name:
                    rand_var_key = copy.deepcopy(key)
                    break
        if rand_var_key is None:
            raise Exception('Unable to find {} in {}'.format(dependent_var_name, summary_dicts[list(summary_dicts.keys())[0]].keys()))

        # get subjects, rand vars and (optional) error (e.g. std) for each var 
        subjects = []
        rand_vars = []
        errors = []
        if control_var_name != 'Load':
            # must append control var param values to list
            params = []
            params_iterator = list(summary_dicts.keys())
        else:
            # control var param values are just the loads
            params = [load for load in summary_dicts[0]['Load'].values() if round(load, 1) in loads] 
            params_iterator = copy.deepcopy(np.unique(params))
        for param in params_iterator:
            if control_var_name != 'Load':
                all_subjects = list(summary_dicts[param]['Subject'].values())
                all_rand_vars = list(summary_dicts[param][rand_var_key].values())
                # num entries for this load
                num_entries = list(summary_dicts[list(summary_dicts.keys())[0]]['Load'].values()).count(load)
                # index where entries for this load start
                starting_idx = list(summary_dicts[list(summary_dicts.keys())[0]]['Load'].values()).index(load)
                # get errors
                if error_summary_dicts is not None:
                    all_errors = list(error_summary_dicts[param][rand_var_key].values())
            else:
                # param is the load
                all_subjects = list(summary_dicts[0]['Subject'].values())
                all_rand_vars = list(summary_dicts[0][rand_var_key].values())
                # num entries for this load
                num_entries = list(summary_dicts[0]['Load'].values()).count(param)
                # index where entries for this load start
                starting_idx = list(summary_dicts[0]['Load'].values()).index(param)
                # get errors
                if error_summary_dicts is not None:
                    all_errors = list(error_summary_dicts[0][rand_var_key].values())

            for idx in range(starting_idx, starting_idx+num_entries):
                subject, rand_var = all_subjects[idx], all_rand_vars[idx]
                if subject in subjects_to_plot:
                    rand_vars.append(all_rand_vars[idx])
                    subjects.append(all_subjects[idx])
                    if error_summary_dicts is not None:
                        errors.append(all_errors[idx])
                    if control_var_name != 'Load':
                        params.append(param)

        # organise into plot_dict
        if error_summary_dicts is None:
            plot_dict = {_class: {'x_values': [], 'y_values': [], 'rand_vars': []} for _class in np.unique(subjects)}
        else:
            plot_dict = {_class: {'x_values': [], 'y_values': [], 'rand_vars': [], 'errors': []} for _class in np.unique(subjects)}
        # for subject, rand_var, param in zip(subjects, rand_vars, params):
        for idx in range(len(subjects)):
            plot_dict[subjects[idx]]['x_values'].append(float(params[idx]))
            plot_dict[subjects[idx]]['y_values'].append(rand_vars[idx])
            plot_dict[subjects[idx]]['rand_vars'].append(rand_vars[idx])
            if error_summary_dicts is not None:
                plot_dict[subjects[idx]]['errors'].append(errors[idx])

        if kwargs['replace_summary_dict_dependent_var_name']:
            rand_var_key = copy.deepcopy(dependent_var_display_name)

        if control_var_name != 'Load':
            title = 'Load {}'.format(load)
        else:
            title = None

        error_bar_axis = None
    
        # scatter
        if error_summary_dicts is not None:
            error_bar_axis = 'yerr'
        fig1 = plot_dists.plot_val_scatter(plot_dict=plot_dict, 
                                           xlabel=control_var_name, 
                                           # ylabel='Load {} '.format(load)+rand_var_key,
                                           ylabel=dependent_var_display_name,
                                           title=title,
                                           ghost_classes=ghost_subjects,
                                           plot_line=True,
                                           aspect=kwargs['aspect'],
                                           error_bar_axis=error_bar_axis,
                                           gridlines=kwargs['gridlines'],
                                           use_scientific_notation_yaxis=kwargs['use_scientific_notation'],
                                           figsize=kwargs['scatter_figsize'],
                                           font_size=kwargs['font_size'],
                                           ylogscale=kwargs['logscale'],
                                           legend_ncol=kwargs['legend_ncol'],
                                           show_fig=True)

        # complementary cdf
        fig2 = plot_dists.plot_val_cdf(plot_dict=plot_dict, 
                                       xlabel=dependent_var_display_name, 
                                       ylabel='C-CDF', 
                                       title=title,
                                       ghost_classes=ghost_subjects,
                                       complementary_cdf=True, 
                                       aspect=kwargs['aspect'],
                                       use_scientific_notation=kwargs['use_scientific_notation'],
                                       gridlines=kwargs['gridlines'],
                                       logscale=kwargs['logscale'],
                                       figsize=kwargs['cdf_figsize'],
                                       font_size=kwargs['font_size'],
                                       legend_ncol=kwargs['legend_ncol'],
                                       show_fig=True)

        # % change
        for key in plot_dict.keys():
            init_idx = np.argmin(plot_dict[key]['x_values'])
            init_var = plot_dict[key]['y_values'][init_idx]
            plot_dict[key]['y_values'] = [plot_dict[key]['y_values'][i]/init_var for i in range(len(plot_dict[key]['y_values']))]
        if '(' in dependent_var_display_name:
            # remove units since this y-axis label will be a fraction
            _dependent_var_display_name = dependent_var_display_name.split(' (')[0]
        else:
            _dependent_var_display_name = dependent_var_display_name
        fig3 = plot_dists.plot_val_scatter(plot_dict=plot_dict, 
                                           xlabel=control_var_name, 
                                           # ylabel='Load {} '.format(load)+'\u0394F {}'.format(dependent_var_name), 
                                           ylabel='\u0394F {}'.format(_dependent_var_display_name), 
                                           title=title,
                                           error_bar_axis=None, # cant do error for % change since calc from mean
                                           ghost_classes=ghost_subjects,
                                           gridlines=kwargs['gridlines'], 
                                           aspect=kwargs['aspect'], 
                                           plot_line=True,
                                           figsize=kwargs['scatter_figsize'], 
                                           use_scientific_notation_yaxis=False,
                                           font_size=kwargs['font_size'],
                                           ylogscale=False,
                                           legend_ncol=kwargs['legend_ncol'], 
                                           show_fig=True)

        if path_to_save is not None:
            if '/' in rand_var_key:
                # change so that save name is compatible
                idx = rand_var_key.index('/')
                rand_var_key = rand_var_key[:idx] + '_' + rand_var_key [idx+1:]
            if control_var_name != 'Load':
                fig1.savefig(path_to_save+'/{}_vs_{}_load_{}_scatter.png'.format(rand_var_key, control_var_name, load), bbox_inches='tight')
                fig2.savefig(path_to_save+'/{}_vs_{}_load_{}_complementary_cdf.png'.format(rand_var_key, control_var_name, load), bbox_inches='tight')
                fig3.savefig(path_to_save+'/{}_vs_{}_load_{}_change.png'.format(rand_var_key, control_var_name, load), bbox_inches='tight')
            else:
                fig1.savefig(path_to_save+'/{}_vs_{}_scatter.png'.format(rand_var_key, control_var_name), bbox_inches='tight')
                fig2.savefig(path_to_save+'/{}_vs_{}_complementary_cdf.png'.format(rand_var_key, control_var_name), bbox_inches='tight')
                fig3.savefig(path_to_save+'/{}_vs_{}_change.png'.format(rand_var_key, control_var_name), bbox_inches='tight')
            print('Saved figs to {}'.format(path_to_save))

        counter += 1


    return [fig1, fig2, fig3]





