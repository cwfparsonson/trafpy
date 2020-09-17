from trafpy.generator.src.networks import gen_fat_tree

import numpy as np


# define benchmark version
BENCHMARK_VERSION = '0.0.1'

# define benchmarks to generate
BENCHMARKS = ['university', 'private_enterprise', 'commercial_cloud', 'social_media_cloud']

# define network topology for each benchmark
NETS = {'university': gen_fat_tree(k=4, N=30, num_channels=1),
        'private_enterprise': gen_fat_tree(k=6, N=30, num_channels=1),
        'commercial_cloud': gen_fat_tree(k=11, N=30, num_channels=1),
        'social_media_cloud': gen_fat_tree(k=23, N=35, num_channels=1)}

# define network capacity for each benchmark
NETWORK_CAPACITIES = {'university': NETS['university'].graph['max_nw_capacity'],
                      'private_enterprise': NETS['private_enterprise'].graph['max_nw_capacity'],
                      'commercial_cloud': NETS['commercial_cloud'].graph['max_nw_capacity'],
                      'social_media_cloud': NETS['social_media_cloud'].graph['max_nw_capacity']}

# define network racks for each benchmark
RACKS_DICTS = {'university': NETS['university'].graph['rack_to_ep_dict'],
               'private_enterprise': NETS['private_enterprise'].graph['rack_to_ep_dict'],
               'commercial_cloud': NETS['commercial_cloud'].graph['rack_to_ep_dict'],
               'social_media_cloud': NETS['social_media_cloud'].graph['rack_to_ep_dict']}


# define network load fractions
LOADS = np.arange(0.1, 1.1, 0.1).tolist()
LOADS = [round(load, 3) for load in LOADS] # ensure no python floating point arithmetic errors

# define number of repetitions to perform for each benchmark for each load
NUM_REPEATS = 1

assert len(BENCHMARKS) == len(NETS.keys()) == len(NETWORK_CAPACITIES.keys()) == len(RACKS_DICTS.keys()), \
    'Must specify BENCHMARKS, NETS, NETWORK_CAPACITIES and RACKS_DICTS for each benchmark specified.'



