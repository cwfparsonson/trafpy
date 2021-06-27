import trafpy
from trafpy.generator.src.tools import load_data_from_json, save_data_as_json

import abc
import os
from pathlib import Path
import json
import numpy as np

def get_default_benchmarks():
    '''
    Gets list of default benchmarks in TrafPy.
    '''
    default_benchmarks = ['university',
                          'private_enterprise',
                          'commercial_cloud',
                          'social_media_cloud',
                          'uniform',
                           'skewed_nodes_sensitivity_0', 
                           'skewed_nodes_sensitivity_0.05', 
                           'skewed_nodes_sensitivity_0.1',
                           'skewed_nodes_sensitivity_0.2',
                           'skewed_nodes_sensitivity_0.4',
                           'rack_dist_sensitivity_0',
                           'rack_dist_sensitivity_0.2',
                           'rack_dist_sensitivity_0.4',
                           'rack_dist_sensitivity_0.6',
                           'rack_dist_sensitivity_0.8',
                           'jobcentric_prototyping',
                           'tensorflow']
    return default_benchmarks 


class Benchmark(abc.ABC):
    def __init__(self, benchmark_name, benchmark_version='v001', load_prev_dists=True, jobcentric=False):
        self.benchmark_name = benchmark_name
        self.benchmark_version = benchmark_version
        self.load_prev_dists = load_prev_dists
        self.jobcentric = jobcentric

        trafpy_path = os.path.dirname(trafpy.__file__)
        self.benchmark_path = trafpy_path + '/benchmarker/versions/benchmark_{}/data/{}/'.format(benchmark_version, benchmark_name)
        Path(self.benchmark_path).mkdir(exist_ok=True)
        if load_prev_dists:
            print('Set to load benchmark {} distribution data from {}'.format(benchmark_name, self.benchmark_path))
        else:
            print('Set to save benchmark {} distribution data to {}'.format(benchmark_name, self.benchmark_path))

    def load_dist(self, benchmark_name, dist_name):
        # check if dists already previously saved 
        path = self.benchmark_path+'{}.json'.format(dist_name)
        if os.path.exists(path):
            dist = json.loads(load_data_from_json(path_to_load=path, print_times=False))
            if type(dist) == dict:
                # convert keys (rand var unique values) from str to float
                dist = {float(key): dist[key] for key in dist.keys()}
            else:
                # convert list to numpy array
                dist = np.asarray(dist)
            print('Loaded {} distribution data from {}'.format(dist_name, path))
        else:
            dist = None

        return dist, path 

    def get_dist_and_path(self, dist_name):
        dist = None
        if self.load_prev_dists:
            # load dist and get path dist saved in
            dist, path = self.load_dist(self.benchmark_name, dist_name=dist_name)
        else:
            # just get path for saving
            path = self.benchmark_path+'{}.json'.format(dist_name)

        return dist, path

    def save_dist(self, dist_data, dist_name):
        save_data_as_json(path_to_save=self.benchmark_path+'{}.json'.format(dist_name), data=dist_data, overwrite=True, print_times=False)
        print('Saved {} distribution data to {}'.format(dist_name, self.benchmark_path))


    @abc.abstractmethod
    def get_node_dist(self, eps, racks_dict, dist_name='node_dist'):
        dist, path = self.get_dist_and_path(dist_name) 
        if dist is not None:
            if len(eps) != len(dist):
                raise Exception('You provided len(eps)={} end points but the node distribution used has len(node_dist)={} end points. This is likely because you have left load_prev_dists=True but you are now trying to generate traffic for a network with a different number of end points. Set load_prev_dists=False or ensure len(eps) == len(node_dist)'.format(len(eps), len(dist)))
        return dist, path


    @abc.abstractmethod
    def get_interarrival_time_dist(self, dist_name='interarrival_time_dist'):
        dist, path = self.get_dist_and_path(dist_name) 
        return dist, path

    @abc.abstractmethod
    def get_flow_size_dist(self, dist_name='flow_size_dist'):
        dist, path = self.get_dist_and_path(dist_name) 
        return dist, path

    def get_num_ops_dist(self, dist_name='num_ops_dist'):
        if self.jobcentric:
            # must implement this method if jobcentric
            raise NotImplementedError('If jobcentric==True, must implement get_num_ops_dist method')
        else:
            # no need to implement for flow centric
            pass
        dist, path = self.get_dist_and_path(dist_name) 
        return dist, path
