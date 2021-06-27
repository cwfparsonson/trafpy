import trafpy
from trafpy.benchmarker.versions.benchmark_v001 import config
from trafpy.generator.src.dists import node_dists
from trafpy.generator.src.dists import val_dists
from trafpy.generator.src.dists import plot_dists
from trafpy.generator.src.tools import load_data_from_json, save_data_as_json
import os
import json

import matplotlib.pyplot as plt
import numpy as np
import cv2
import io
import copy
import math




class DistributionGenerator:
    def __init__(self, load_prev_dists=True):
        self.load_prev_dists=load_prev_dists
        self.benchmark_version = 'v001'
        self.valid_benchmark_sets = config.ALL_BENCHMARK_SETS

        trafpy_path = os.path.dirname(trafpy.__file__)
        self.benchmark_version_path = trafpy_path + '/benchmarker/versions/benchmark_v001/'

        self.dist_names = ['node_dist', 'interarrival_time_dist', 'flow_size_dist', 'num_ops_dist']



    def init_dir(self, benchmark):
        # check if data folder exists
        if os.path.exists(self.benchmark_version_path+'data'):
            # data folder already exists
            pass
        else:
            print('Creating data folder in {}'.format(self.benchmark_version_path))
            os.mkdir(self.benchmark_version_path+'data')

        # check if benchmark folder exists in data folder
        if os.path.exists(self.benchmark_version_path+'data/'+str(benchmark)):
            # benchmark folder already exists
            pass
        else:
            print('Creating {} benchmark folder in {}'.format(benchmark, self.benchmark_version_path+'data/'))
            os.mkdir(self.benchmark_version_path+'data/'+benchmark)

    def load_dist(self, benchmark, dist_name):
        # check if dists already previously saved 
        path_to_data = self.benchmark_version_path+'data/{}/{}.json'.format(benchmark, dist_name)
        if os.path.exists(path_to_data):
            print('Loading previously saved benchmark dists from {}'.format(path_to_data))
            dist = json.loads(load_data_from_json(path_to_load=path_to_data, print_times=True))
            if type(dist) == dict:
                # convert keys (rand var unique values) from str to float
                dist = {float(key): dist[key] for key in dist.keys()}
            else:
                # convert list to numpy array
                dist = np.asarray(dist)
        else:
            print('{} distribution not previously saved in {}.'.format(dist_name, path_to_data))
            dist = None

        return dist, path_to_data


    def get_node_dist(self, benchmark, racks_dict, eps, save_data=True):
        self.init_dir(benchmark)

        if benchmark not in self.valid_benchmark_sets:
            raise Exception('Unrecognised benchmark set \'{}\'. Valid benchmark sets for benchmark {}:\n{}'.format(benchmark, self.benchmark_version, self.valid_benchmark_sets))

        num_skewed_nodes = math.ceil(0.2 * len(eps))
        skewed_node_probs = [0.55/num_skewed_nodes for _ in range(num_skewed_nodes)]

        dist_name = 'node_dist'
        node_dist = None
        if self.load_prev_dists:
            # attempt to load previously saved dist
            node_dist, path_to_data = self.load_dist(benchmark, dist_name=dist_name)
        else:
            # just get path_to_data for saving
            path_to_data = self.benchmark_version_path+'data/{}/{}.json'.format(benchmark, dist_name)

        if node_dist is None or not self.load_prev_dists:
            print('Generating {} distribution for {} benchmark...'.format(dist_name, benchmark))

            if benchmark == 'university':
                num_skewed_nodes = math.ceil(0.2 * len(eps))
                skewed_node_probs = [0.55/num_skewed_nodes for _ in range(num_skewed_nodes)]
                if racks_dict is None:
                    rack_prob_config = None
                else:
                    rack_prob_config = {'racks_dict': racks_dict, 'prob_inter_rack': 0.7}
                # node_dist = node_dists.gen_uniform_node_dist(eps, rack_prob_config=rack_prob_config, show_fig=False, print_data=False)
                node_dist = node_dists.gen_multimodal_node_dist(eps, 
                                                                rack_prob_config=rack_prob_config, 
                                                                num_skewed_nodes=num_skewed_nodes, 
                                                                skewed_node_probs=skewed_node_probs, 
                                                                show_fig=False, 
                                                                print_data=False)

            else:
                raise Exception('Benchmark \'{}\' not recognised.'.format(benchmark))




            if save_data:
                save_data_as_json(path_to_save=path_to_data, data=node_dist, overwrite=True, print_times=False)

        # check node dist is compatible with eps provided
        if len(eps) != len(node_dist):
            raise Exception('You provided len(eps)={} end points but the node distribution used has len(node_dist)={} end points. This is likely because you have left load_prev_dists=True but you are now trying to generate traffic for a network with a different number of end points. Set load_prev_dists=False or ensure len(eps) == len(node_dist)'.format(len(eps), len(node_dist)))


        return node_dist


    def get_flow_size_dist(self, benchmark, save_data=True):
        self.init_dir(benchmark)

        if benchmark not in self.valid_benchmark_sets:
            raise Exception('Unrecognised benchmark set \'{}\'. Valid benchmark sets for benchmark {}:\n{}'.format(benchmark, self.benchmark_version, self.valid_benchmark_sets))

        dist_name = 'flow_size_dist'
        flow_size_dist = None
        if self.load_prev_dists:
            # attempt to load previously saved dist
            flow_size_dist, path_to_data = self.load_dist(benchmark, dist_name=dist_name)
        else:
            # just get path_to_data for saving
            path_to_data = self.benchmark_version_path+'data/{}/{}.json'.format(benchmark, dist_name)

        if flow_size_dist is None or not self.load_prev_dists:
            print('Generating {} distribution for {} benchmark...'.format(dist_name, benchmark))

            if benchmark == 'university':
                flow_size_dist = val_dists.gen_named_val_dist(dist='lognormal',
                                                              params={'_mu': 7, '_sigma': 2.5},
                                                              min_val=1,
                                                              max_val=2e7,
                                                              round_to_nearest=25,
                                                              show_fig=False,
                                                              print_data=False)
            else:
                raise Exception('Benchmark \'{}\' not recognised.'.format(benchmark))


            if save_data:
                save_data_as_json(path_to_save=path_to_data, data=flow_size_dist, overwrite=True, print_times=False)



        return flow_size_dist




    def get_interarrival_time_dist(self, benchmark, save_data=True):
        self.init_dir(benchmark)

        if benchmark not in self.valid_benchmark_sets:
            raise Exception('Unrecognised benchmark set \'{}\'. Valid benchmark sets for benchmark {}:\n{}'.format(benchmark, self.benchmark_version, self.valid_benchmark_sets))

        dist_name = 'interarrival_time_dist'
        interarrival_time_dist = None
        if self.load_prev_dists:
            # attempt to load previously saved dist
            interarrival_time_dist, path_to_data = self.load_dist(benchmark, dist_name=dist_name)
        else:
            # just get path_to_data for saving
            path_to_data = self.benchmark_version_path+'data/{}/{}.json'.format(benchmark, dist_name)

        if interarrival_time_dist is None or not self.load_prev_dists:
            print('Generating {} distribution for {} benchmark...'.format(dist_name, benchmark))

            if benchmark == 'university':
                interarrival_time_dist = val_dists.gen_named_val_dist(dist='weibull',
                                                                      params={'_alpha': 0.9, '_lambda': 6000},
                                                                      min_val=1,
                                                                      round_to_nearest=25,
                                                                      show_fig=False,
                                                                      print_data=False)

            else:
                raise Exception('Benchmark \'{}\' not recognised.'.format(benchmark))


            if save_data:
                save_data_as_json(path_to_save=path_to_data, data=interarrival_time_dist, overwrite=True, print_times=False)


        return interarrival_time_dist


    def get_num_ops_dist(self, benchmark, save_data=True):
        self.init_dir(benchmark)

        if benchmark not in self.valid_benchmark_sets:
            raise Exception('Unrecognised benchmark set \'{}\'. Valid benchmark sets for benchmark {}:\n{}'.format(benchmark, self.benchmark_version, self.valid_benchmark_sets))

        dist_name = 'num_ops_dist'
        num_ops_dist = None
        if benchmark not in ['jobcentric_prototyping',
                             'tensorflow']:
            # not a job-centric benchmark, no num_ops_dist, return None
            return None

        else:
            # job-centric benchmark, get num_ops_dist
            if self.load_prev_dists:
                # attempt to load previously saved dist
                num_ops_dist, path_to_data = self.load_dist(benchmark, dist_name=dist_name)
            else:
                # just get path_to_data for saving
                path_to_data = self.benchmark_version_path+'data/{}/{}.json'.format(benchmark, dist_name)

        if num_ops_dist is None or not self.load_prev_dists:
            print('Generating {} distribution for {} benchmark...'.format(dist_name, benchmark))

            if benchmark == 'jobcentric_prototyping':
                num_ops_dist = {10: 1}

            elif benchmark == 'tensorflow':
                # num_ops_dist = val_dists.gen_named_val_dist(dist='lognormal',
                                                              # params={'_mu': 4.55, '_sigma': 0.18},
                                                              # min_val=50,
                                                              # max_val=200,
                                                              # round_to_nearest=1,
                                                              # show_fig=False,
                                                              # print_data=False)
                num_ops_dist = val_dists.gen_named_val_dist(dist='skewnorm',
                                                            params={'_a': 2.3, '_loc': 80, '_scale': 30},
                                                              min_val=50,
                                                              max_val=200,
                                                              round_to_nearest=10,
                                                              show_fig=False,
                                                              print_data=False)

            else:
                raise Exception('Benchmark \'{}\' not recognised.'.format(benchmark))

        return num_ops_dist

    


    def plot_benchmark_dists(self, benchmarks, fontsize=20, time_units='\u03BCs', info_units='B'):
        '''Plots dist info of all benchmark(s).

        e.g. benchmarks = ['uniform', 'university']

        '''

        for benchmark in benchmarks:
            if benchmark not in self.valid_benchmark_sets:
                raise Exception('Benchmark \'{}\' not recognised. Must be one of: {}'.format(benchmark, self.valid_benchmark_sets))


        # load dists
        dists = {benchmark: {dist_name: None for dist_name in self.dist_names} for benchmark in benchmarks}
        plotted_rand_vars = copy.deepcopy(dists)
        plots = copy.deepcopy(dists)
        for benchmark in benchmarks:
            for dist_name in self.dist_names:

                # load dist
                dists[benchmark][dist_name], path_to_data = self.load_dist(benchmark, dist_name)
                if type(dists[benchmark][dist_name]) is str:
                    dists[benchmark][dist_name] = json.loads(dists[benchmark][dist_name])

                # check loaded dist successfully
                if dists[benchmark][dist_name] is None:
                    print('Dist {} for benchmark {} not found in {}. Ensure dist is named as one of {}, and that dist has been saved in correct location.'.format(dist_name, benchmark, path_to_data, self.dist_names))

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
                        # fig = plot_dists.plot_val_dist(rand_vars, show_fig=True, figsize=(12.4, 2), plot_horizontally=True, logscale=logscale, num_bins=20, rand_var_name='Flow Size ({})'.format(info_units), font_size=fontsize)
                        fig = plot_dists.plot_val_dist(rand_vars, show_fig=True, figsize=(6.2, 4), use_scientific_notation_yaxis=True, plot_horizontally=False, logscale=logscale, num_bins=20, rand_var_name='Flow Size ({})'.format(info_units), font_size=fontsize)
                        plots[benchmark][dist_name] = fig
                    elif dist_name == 'interarrival_time_dist':
                        # fig = plot_dists.plot_val_dist(rand_vars, show_fig=True, figsize=(12.4, 2), plot_horizontally=True, logscale=logscale, num_bins=20, rand_var_name='Interarrival Time ({})'.format(time_units), font_size=fontsize)
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



                    
    def conv_fig_to_image(self, fig, dpi=300):
        '''
        Takes matplotlib figure and converts into numpy array of RGB pixel values
        '''
        # minimise whitespace around edges
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        buf = io.BytesIO()
        try:
            fig.savefig(buf, bbox_inches='tight', pad_inches=0, format='png', dpi=dpi)
        except AttributeError:
            fig.figure.savefig(buf, bbox_inches='tight', pad_inches=0, format='png', dpi=dpi)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr,1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img









        






