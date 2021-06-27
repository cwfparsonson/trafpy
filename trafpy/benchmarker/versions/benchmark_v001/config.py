import trafpy
from trafpy.generator.src.dists import node_dists
from trafpy.generator.src.dists import val_dists
from trafpy.generator.src.dists import plot_dists

import os
import glob
import copy
import importlib
import json


def get_default_benchmark_names():
    '''
    Gets list of default benchmarks in TrafPy.
    '''
    trafpy_path = os.path.dirname(trafpy.__file__)
    path_to_benchmarks = trafpy_path + '/benchmarker/versions/benchmark_v001/benchmarks/'
    paths = glob.glob(path_to_benchmarks + '*.py')
    return sorted([os.path.basename(path).split('.')[0] for path in paths])

def plot_benchmark_dists(benchmark_names, fontsize=20, time_units='\u03BCs', info_units='B'):
    '''Plots dist info of all benchmark(s).

    e.g. benchmark_names = ['uniform', 'university']

    '''

    for benchmark in benchmark_names:
        if benchmark not in get_default_benchmark_names():
            raise Exception('Benchmark \'{}\' not recognised. Must be one of: {}'.format(benchmark, get_default_benchmark_names()))


    # load dists
    dist_names = ['node_dist', 'interarrival_time_dist', 'flow_size_dist', 'num_ops_dist']
    dists = {benchmark: {dist_name: None for dist_name in dist_names} for benchmark in benchmark_names}
    plotted_rand_vars = copy.deepcopy(dists)
    plots = copy.deepcopy(dists)
    for benchmark in benchmark_names:
        print('\n~* {} *~'.format(benchmark))
        # import benchmark class and instantiate benchmark object
        benchmark_module = importlib.import_module('trafpy.benchmarker.versions.benchmark_v001.benchmarks.{}'.format(benchmark))
        b = benchmark_module.DefaultBenchmark(benchmark_name=benchmark, benchmark_version='v001', load_prev_dists=True)

        for dist_name in dist_names:

            # load dist
            dists[benchmark][dist_name], path = b.load_dist(benchmark, dist_name)
            if type(dists[benchmark][dist_name]) is str:
                dists[benchmark][dist_name] = json.loads(dists[benchmark][dist_name])

            # check loaded dist successfully
            if dists[benchmark][dist_name] is None:
                print('Dist {} for benchmark {} not found in {}. Ensure dist is named as one of {}, and that dist has been saved in correct location.'.format(dist_name, benchmark, path, get_default_benchmark_names()))

            num_demands = max(len(dists[benchmark]['node_dist']) * 1000, 200000) # estimate appropriate number of rand vars to gen 

            if dist_name in ['flow_size_dist', 'interarrival_time_dist']:
                # remove str keys
                dists[benchmark][dist_name] = {float(key): val for key, val in dists[benchmark][dist_name].items()}

                # generate random variables from dist to plot
                rand_vars = val_dists.gen_rand_vars_from_discretised_dist(unique_vars=list(dists[benchmark][dist_name].keys()),
                                                                            probabilities=list(dists[benchmark][dist_name].values()),
                                                                            num_demands=num_demands)
                plotted_rand_vars[benchmark][dist_name] = rand_vars

                if all(prob == list(dists[benchmark][dist_name].values())[0] for prob in dists[benchmark][dist_name].values()):
                    # uniform dist, do not plot logscale
                    logscale = False
                else:
                    logscale = True

                if dist_name == 'flow_size_dist':
                    fig = plot_dists.plot_val_dist(rand_vars, show_fig=True, figsize=(6.2, 4), use_scientific_notation_yaxis=True, plot_horizontally=False, logscale=logscale, num_bins=20, rand_var_name='Flow Size ({})'.format(info_units), font_size=fontsize)
                    plots[benchmark][dist_name] = fig
                elif dist_name == 'interarrival_time_dist':
                    fig = plot_dists.plot_val_dist(rand_vars, show_fig=True, figsize=(6.2, 4), use_scientific_notation_yaxis=True, plot_horizontally=False, logscale=logscale, num_bins=20, rand_var_name='Interarrival Time ({})'.format(time_units), font_size=fontsize)
                    plots[benchmark][dist_name] = fig

            elif dist_name == 'node_dist':
                fig = plot_dists.plot_node_dist(dists[benchmark][dist_name],
                                                chord_edge_width_range=[1,25],
                                                chord_edge_display_threshold=0.35,
                                                font_size=fontsize,
                                                show_fig=True) # 0.475
                plots[benchmark][dist_name] = fig

            else:
                print('Unrecognised dist_name {}'.format(dist_name))
                
    
    
    return plots, dists, plotted_rand_vars


