import trafpy
from trafpy.generator.src.tools import load_data_from_json, save_data_as_json

import abc
import os
from pathlib import Path
import json
import numpy as np


class Benchmark(abc.ABC):
    def __init__(self, benchmark_name, benchmark_version='v001', load_prev_dists=True, jobcentric=False):
        '''
        Args:
            benchmark_name (str): Name of benchmark (e.g. 'university') 
            benchmark_version (str): TrafPy benchmark version to access (e.g. 'v001').
            load_prev_dists (bool): If True, will generate a new benchmark distribution
                for the network(s) you provide the imported. This is needed if
                you have a network with a different number of endpoints or with
                different end point labels. If False, will load the exact same distributions as was previously
                defined, which is needed if you want to use the exact same benchmark
                distribution multiple times.
            jobcentric (bool): Whether or not the benchmark traffic is job-centric (True)
                or flow-centric (False).
        '''
        self.benchmark_name = benchmark_name
        self.benchmark_version = benchmark_version
        self.load_prev_dists = load_prev_dists
        self.jobcentric = jobcentric

        trafpy_path = os.path.dirname(trafpy.__file__)
        # self.benchmark_path = trafpy_path + '/benchmarker/versions/benchmark_{}/data/{}/'.format(benchmark_version, benchmark_name)
        self.benchmark_path = trafpy_path + '/benchmarker/versions/benchmark_{}/benchmarks/{}/'.format(benchmark_version, benchmark_name)
        Path(self.benchmark_path).mkdir(exist_ok=True)
        if load_prev_dists:
            print('Set to load benchmark {} distribution data from {}'.format(benchmark_name, self.benchmark_path))
        else:
            print('Set to save benchmark {} distribution data to {}'.format(benchmark_name, self.benchmark_path))

    def load_dist(self, benchmark_name, dist_name):
        '''Loads previously saved distribution data for given benchmark.'''
        # check if dists already previously saved 
        path = self.benchmark_path+'{}.json'.format(dist_name)
        if os.path.exists(path):
            dist_data = json.loads(load_data_from_json(path_to_load=path, print_times=False))
            if type(dist_data) == dict:
                # convert keys (rand var unique values) from str to float
                dist_data = {float(key): dist_data[key] for key in dist_data.keys()}
            else:
                # convert list to numpy array
                dist_data = np.asarray(dist_data)
            print('Loaded {} distribution data from {}'.format(dist_name, path))
        else:
            dist_data = None

        return dist_data, path 

    def get_dist_and_path(self, dist_name):
        '''Gets distribution data and corresponding path.

        If distribution data does not exist, will return dist=None and the path
        will be where the dist data should be saved if it is generated.
        '''
        dist_data = None
        if self.load_prev_dists:
            # load dist and get path dist saved in
            dist_data, path = self.load_dist(self.benchmark_name, dist_name=dist_name)
        else:
            # just get path for saving
            path = self.benchmark_path+'{}.json'.format(dist_name)

        return dist_data, path

    def save_dist(self, dist_data, dist_name):
        '''Saves distribution data for a given benchmark.'''
        save_data_as_json(path_to_save=self.benchmark_path+'{}.json'.format(dist_name), data=dist_data, overwrite=True, print_times=False)
        print('Saved {} distribution data to {}'.format(dist_name, self.benchmark_path))


    @abc.abstractmethod
    def get_node_dist(self, eps, racks_dict, dist_name='node_dist'):
        '''Loads previously saved node dist (if it exists).

        This is an abstract method and therefore must be defined by
        any child class.

        Args:
            eps (list): List of network end points.
            racks_dict (dict): Dict mapping racks to the corresponding end points
                contained within each rack.
            dist_name (str): Name of distribution (determines path to search for
                previously saved distribution).
        '''
        dist, path = self.get_dist_and_path(dist_name) 
        if dist is not None:
            if len(eps) != len(dist):
                raise Exception('You provided len(eps)={} end points but the node distribution used has len(node_dist)={} end points. This is likely because you have left load_prev_dists=True but you are now trying to generate traffic for a network with a different number of end points. Set load_prev_dists=False or ensure len(eps) == len(node_dist)'.format(len(eps), len(dist)))
        return dist, path


    @abc.abstractmethod
    def get_interarrival_time_dist(self, dist_name='interarrival_time_dist'):
        '''Loads previously saved interarrival time dist (if it exists).

        This is an abstract method and therefore must be defined by
        any child class.

        Args:
            dist_name (str): Name of distribution (determines path to search for
                previously saved distribution).
        '''
        dist, path = self.get_dist_and_path(dist_name) 
        return dist, path

    @abc.abstractmethod
    def get_flow_size_dist(self, dist_name='flow_size_dist'):
        '''Loads previously saved flow size dist (if it exists).

        This is an abstract method and therefore must be defined by
        any child class.

        Args:
            dist_name (str): Name of distribution (determines path to search for
                previously saved distribution).
        '''
        dist, path = self.get_dist_and_path(dist_name) 
        return dist, path

    def get_num_ops_dist(self, dist_name='num_ops_dist'):
        '''Loads previously saved number of operations dist (if it exists).

        This method only needs to be defined by a child class if jobcentric=True,
        since flow-centric data have no notion of 'number of operations' (in relation
        to job DAGs).

        Args:
            dist_name (str): Name of distribution (determines path to search for
                previously saved distribution).
        '''
        if self.jobcentric:
            # must implement this method if jobcentric
            raise NotImplementedError('If jobcentric==True, must implement get_num_ops_dist method')
        else:
            # no need to implement for flow centric
            pass
        dist, path = self.get_dist_and_path(dist_name) 
        return dist, path
