from trafpy.benchmarker.versions.benchmark import Benchmark
from trafpy.generator.src.dists import node_dists
from trafpy.generator.src.dists import val_dists
from trafpy.generator.src.dists import plot_dists

import math
import numpy as np

class DefaultBenchmark(Benchmark):
    def __init__(self, benchmark_name='jobcentric_prototyping', benchmark_version='v001', load_prev_dists=True, jobcentric=True):
        super(DefaultBenchmark, self).__init__(benchmark_name, benchmark_version, load_prev_dists)

    def get_node_dist(self, eps, racks_dict=None, dist_name='node_dist'):
        dist, path = super().get_node_dist(eps, racks_dict, dist_name)
        if dist is None or not self.load_prev_dists:
            dist = node_dists.gen_uniform_node_dist(eps, show_fig=False, print_data=False)
            super().save_dist(dist, dist_name)
        return dist

    def get_interarrival_time_dist(self, dist_name='interarrival_time_dist'):
        dist, path = super().get_interarrival_time_dist(dist_name)
        if dist is None or not self.load_prev_dists:
            dist = {10000.0: 1.0}
            super().save_dist(dist, dist_name)
        return dist

    def get_flow_size_dist(self, dist_name='flow_size_dist'):
        dist, path = super().get_flow_size_dist(dist_name)
        if dist is None or not self.load_prev_dists:
            dist = {10.0: 1}
            super().save_dist(dist, dist_name)
        return dist

    def get_num_ops_dist(self, dist_name='num_ops_dist'):
        dist, path = super().get_flow_size_dist(dist_name)
        if dist is None or not self.load_prev_dists:
            dist = {10: 1}
            super().save_dist(dist, dist_name)
        return dist

