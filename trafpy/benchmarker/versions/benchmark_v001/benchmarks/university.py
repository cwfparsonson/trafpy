from trafpy.benchmarker.versions.benchmark import Benchmark
from trafpy.generator.src.dists import node_dists
from trafpy.generator.src.dists import val_dists
from trafpy.generator.src.dists import plot_dists

import math
import numpy as np

class DefaultBenchmark(Benchmark):
    def __init__(self, benchmark_name='university', benchmark_version='v001', load_prev_dists=True):
        super(DefaultBenchmark, self).__init__(benchmark_name, benchmark_version, load_prev_dists)

    def get_node_dist(self, eps, racks_dict=None, dist_name='node_dist'):
        dist, path = super().get_node_dist(eps, racks_dict, dist_name)
        if dist is None or not self.load_prev_dists:
            num_skewed_nodes = math.ceil(0.2 * len(eps))
            skewed_node_probs = [0.55/num_skewed_nodes for _ in range(num_skewed_nodes)]
            if racks_dict is None:
                rack_prob_config = None
            else:
                rack_prob_config = {'racks_dict': racks_dict, 'prob_inter_rack': 0.7}
            dist = node_dists.gen_multimodal_node_dist(eps, 
                                                        rack_prob_config=rack_prob_config, 
                                                        num_skewed_nodes=num_skewed_nodes, 
                                                        skewed_node_probs=skewed_node_probs, 
                                                        show_fig=False, 
                                                        print_data=False)
            super().save_dist(dist, dist_name)
        return dist

    def get_interarrival_time_dist(self, dist_name='interarrival_time_dist'):
        dist, path = super().get_interarrival_time_dist(dist_name)
        if dist is None or not self.load_prev_dists:
            dist = val_dists.gen_named_val_dist(dist='weibull',
                                                params={'_alpha': 0.9, '_lambda': 6000},
                                                min_val=1,
                                                round_to_nearest=25,
                                                show_fig=False,
                                                print_data=False)
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
