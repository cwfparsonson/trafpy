import trafpy
from trafpy.benchmarker.versions.benchmark_v001 import config
from trafpy.generator.src.dists import node_dists
from trafpy.generator.src.dists import val_dists
from trafpy.generator.src.tools import load_data_from_json, save_data_as_json
import os
import json


class DistributionGenerator:
    def __init__(self, load_prev_dists=True):
        self.load_prev_dists=load_prev_dists
        self.benchmark_version = '0.0.1'
        self.valid_benchmark_sets = config.ALL_BENCHMARK_SETS

        trafpy_path = os.path.dirname(trafpy.__file__)
        self.benchmark_version_path = trafpy_path + '/benchmarker/versions/benchmark_v001/'




    def load_dist(self, benchmark, dist_name):

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

        # check if dists already previously saved 
        path_to_data = self.benchmark_version_path+'data/{}/{}.json'.format(benchmark, dist_name)
        if os.path.exists(path_to_data):
            print('Loading previously saved benchmark dists from {}'.format(path_to_data))
            dist = load_data_from_json(path_to_load=path_to_data, print_times=True)
        else:
            print('{} distribution not previously saved in {}.'.format(dist_name, path_to_data))
            dist = None

        return dist, path_to_data


    def get_node_dist(self, benchmark, racks_dict, eps, save_data=True):
        if benchmark not in self.valid_benchmark_sets:
            raise Exception('Unrecognised benchmark set \'{}\'. Valid benchmark sets for benchmark {}:\n{}'.format(benchmark, self.benchmark_version, self.valid_benchmark_sets))

        dist_name = 'node_dist'
        node_dist = None
        if self.load_prev_dists:
            # attempt to load previously saved dist
            node_dist, path_to_data = self.load_dist(benchmark, dist_name=dist_name)

        if node_dist is None or not self.load_prev_dists:
            print('Generating {} distribution for {} benchmark...'.format(dist_name, benchmark))

            if benchmark == 'university':
                rack_prob_config = {'racks_dict': racks_dict, 'prob_inter_rack': 0.7}
                node_dist = node_dists.gen_uniform_node_dist(eps, rack_prob_config=rack_prob_config, show_fig=False, print_data=False)


            if save_data:
                save_data_as_json(path_to_save=path_to_data, data=node_dist, overwrite=True, print_times=False)

        return node_dist


    def get_flow_size_dist(self, benchmark, save_data=True):
        if benchmark not in self.valid_benchmark_sets:
            raise Exception('Unrecognised benchmark set \'{}\'. Valid benchmark sets for benchmark {}:\n{}'.format(benchmark, self.benchmark_version, self.valid_benchmark_sets))

        dist_name = 'flow_size_dist'
        flow_size_dist = None
        if self.load_prev_dists:
            # attempt to load previously saved dist
            flow_size_dist, path_to_data = self.load_dist(benchmark, dist_name=dist_name)

        if flow_size_dist is None or not self.load_prev_dists:
            print('Generating {} distribution for {} benchmark...'.format(dist_name, benchmark))

            if benchmark == 'university':
                flow_size_dist = val_dists.gen_named_val_dist(dist='lognormal',
                                                              params={'_mu': 7, '_sigma': 2.5},
                                                              show_fig=False,
                                                              print_data=False,
                                                              round_to_nearest=1)


            if save_data:
                save_data_as_json(path_to_save=path_to_data, data=flow_size_dist, overwrite=True, print_times=False)



        return flow_size_dist




    def get_interarrival_time_dist(self, benchmark, save_data=True):
        if benchmark not in self.valid_benchmark_sets:
            raise Exception('Unrecognised benchmark set \'{}\'. Valid benchmark sets for benchmark {}:\n{}'.format(benchmark, self.benchmark_version, self.valid_benchmark_sets))

        dist_name = 'interarrival_time_dist'
        interarrival_time_dist = None
        if self.load_prev_dists:
            # attempt to load previously saved dist
            interarrival_time_dist, path_to_data = self.load_dist(benchmark, dist_name=dist_name)

        if interarrival_time_dist is None or not self.load_prev_dists:
            print('Generating {} distribution for {} benchmark...'.format(dist_name, benchmark))

            if benchmark == 'university':
                interarrival_time_dist = val_dists.gen_named_val_dist(dist='weibull',
                                                                      params={'_alpha': 0.9, '_lambda': 6000},
                                                                      show_fig=False,
                                                                      print_data=False,
                                                                      round_to_nearest=1)


            if save_data:
                save_data_as_json(path_to_save=path_to_data, data=interarrival_time_dist, overwrite=True, print_times=False)


        return interarrival_time_dist






