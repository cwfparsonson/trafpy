from trafpy.benchmarker.versions.benchmark import Benchmark
from trafpy.generator.src.dists import node_dists
from trafpy.generator.src.dists import val_dists
from trafpy.generator.src.dists import plot_dists

import math
import numpy as np

class DefaultBenchmark(Benchmark):
    def __init__(self, benchmark_name='rack_sensitivity_02', benchmark_version='v001', load_prev_dists=True):
        super(DefaultBenchmark, self).__init__(benchmark_name, benchmark_version, load_prev_dists)

    def get_node_dist(self, eps, racks_dict=None, dist_name='node_dist'):
        dist, path = super().get_node_dist(eps, racks_dict, dist_name)
        if dist is None or not self.load_prev_dists:
            rack_prob_config = {'racks_dict': racks_dict, 'prob_inter_rack': 0.8} 
            dist = node_dists.gen_uniform_node_dist(eps, 
                                                     rack_prob_config=rack_prob_config, 
                                                     show_fig=False, 
                                                     print_data=False)
            super().save_dist(dist, dist_name)
        return dist

    def get_interarrival_time_dist(self, dist_name='interarrival_time_dist'):
        dist, path = super().get_interarrival_time_dist(dist_name)
        if dist is None or not self.load_prev_dists:
            dist = val_dists.gen_multimodal_val_dist(min_val=1,
                                                   max_val=10000,
                                                   locations=[10,20,100,1],
                                                   skews=[0,0,0,100],
                                                   scales=[1,3,4,50],
                                                   num_skew_samples=[10000,7000,5000,20000],
                                                   round_to_nearest=25,
                                                   bg_factor=0.01)
            super().save_dist(dist, dist_name)
        return dist

    def get_flow_size_dist(self, dist_name='flow_size_dist'):
        dist, path = super().get_flow_size_dist(dist_name)
        if dist is None or not self.load_prev_dists:
            dist = val_dists.gen_named_val_dist(dist='lognormal',
                                              params={'_mu': 7, '_sigma': 2.5},
                                              min_val=1,
                                              max_val=2e7,
                                              round_to_nearest=25,
                                              show_fig=False,
                                              print_data=False)
            super().save_dist(dist, dist_name)
        return dist
