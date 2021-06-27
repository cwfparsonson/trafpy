from trafpy.benchmarker.versions.benchmark import Benchmark
from trafpy.generator.src.dists import node_dists
from trafpy.generator.src.dists import val_dists
from trafpy.generator.src.dists import plot_dists

import math
import numpy as np

class DefaultBenchmark(Benchmark):
    def __init__(self, benchmark_name='tensorflow', benchmark_version='v001', load_prev_dists=True, jobcentric=True):
        super(DefaultBenchmark, self).__init__(benchmark_name, benchmark_version, load_prev_dists)

    def get_node_dist(self, eps, racks_dict=None, dist_name='node_dist'):
        dist, path = super().get_node_dist(eps, racks_dict, dist_name)
        if dist is None or not self.load_prev_dists:
            num_skewed_nodes = math.ceil(0.2 * len(eps))
            skewed_node_probs = [0.55/num_skewed_nodes for _ in range(num_skewed_nodes)]
            if racks_dict is None:
                rack_prob_config = None
            else:
                rack_prob_config = {'racks_dict': racks_dict, 'prob_inter_rack': 0.2}
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
            dist = val_dists.gen_multimodal_val_dist(1,
                                                   1e8,
                                                   locations=[1, 1, 3000, 1, 1800000, 10000000],
                                                   skews=[0, 100, -10, 10, 50, 6],
                                                   scales=[0.1, 62, 2000, 7500, 3500000, 20000000],
                                                   num_skew_samples=[800, 1000, 2000, 4000, 4000, 3000],
                                                   bg_factor=0.025)
            super().save_dist(dist, dist_name)
        return dist

    def get_flow_size_dist(self, dist_name='flow_size_dist'):
        dist, path = super().get_flow_size_dist(dist_name)
        if dist is None or not self.load_prev_dists:
            # taken from DeepMind paper https://arxiv.org/pdf/1905.02494.pdf, no units given. Assume units are in MB -> convert to B
            conversion = 1e6
            dist = val_dists.gen_named_val_dist(dist='normal',
                                                  params={'_loc': 50*conversion, '_scale': 10*conversion},
                                                  min_val=1,
                                                  max_val=None,
                                                  round_to_nearest=25,
                                                  show_fig=False,
                                                  print_data=False)
            super().save_dist(dist, dist_name)
        return dist

    def get_num_ops_dist(self, dist_name='num_ops_dist'):
        dist, path = super().get_flow_size_dist(dist_name)
        if dist is None or not self.load_prev_dists:
            dist = val_dists.gen_named_val_dist(dist='skewnorm',
                                                params={'_a': 2.3, '_loc': 80, '_scale': 30},
                                                  min_val=50,
                                                  max_val=200,
                                                  round_to_nearest=10,
                                                  show_fig=False,
                                                  print_data=False)
            super().save_dist(dist, dist_name)
        return dist

