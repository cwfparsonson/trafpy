from trafpy.generator.src.networks import gen_fat_tree, gen_arbitrary_network

import numpy as np





# -------------------------------------------------------------------------
# general configuration
# -------------------------------------------------------------------------
# define benchmark version
BENCHMARK_VERSION = '0.0.1'

# define factor by which to multiply num endpoints by to get num_demands
NUM_DEMANDS_FACTOR = 50

# define minimum time of last demand's arrival (helps define minimum simulation time)
MIN_LAST_DEMAND_ARRIVAL_TIME = 2e6 # units of us 3e6 6e7 6e8 None 3000.0 2e6 2.4e5pulse

# define network load fractions
LOADS = np.arange(0.1, 1.0, 0.1).tolist()
# LOADS = np.array([0.8])
LOADS = [round(load, 3) for load in LOADS] # ensure no python floating point arithmetic errors

# define number of repetitions to perform for each benchmark for each load
NUM_REPEATS = 1

# define whether or not to auto correct invalid node distribution(s)
AUTO_NODE_DIST_CORRECTION = True










# -------------------------------------------------------------------------
# benchmark-specific configuration (uncomment one below)
# -------------------------------------------------------------------------

# # ALL
# # -------------------------------------------------------------------------
# # define benchmarks to generate
# BENCHMARKS = ['university', 'private_enterprise', 'commercial_cloud', 'social_media_cloud']

# # define network topology for each benchmark
# NETS = {'university': gen_fat_tree(k=4, N=30, num_channels=1),
        # 'private_enterprise': gen_fat_tree(k=6, N=30, num_channels=1),
        # 'commercial_cloud': gen_fat_tree(k=11, N=30, num_channels=1),
        # 'social_media_cloud': gen_fat_tree(k=23, N=35, num_channels=1)}

# # define network capacity for each benchmark
# NETWORK_CAPACITIES = {'university': NETS['university'].graph['max_nw_capacity'],
                      # 'private_enterprise': NETS['private_enterprise'].graph['max_nw_capacity'],
                      # 'commercial_cloud': NETS['commercial_cloud'].graph['max_nw_capacity'],
                      # 'social_media_cloud': NETS['social_media_cloud'].graph['max_nw_capacity']}

# # define network racks for each benchmark
# RACKS_DICTS = {'university': NETS['university'].graph['rack_to_ep_dict'],
               # 'private_enterprise': NETS['private_enterprise'].graph['rack_to_ep_dict'],
               # 'commercial_cloud': NETS['commercial_cloud'].graph['rack_to_ep_dict'],
               # 'social_media_cloud': NETS['social_media_cloud'].graph['rack_to_ep_dict']}








# UNIVERSITY
# -------------------------------------------------------------------------
# define benchmarks to generate
BENCHMARKS = ['university']

# define network topology for each benchmark
# NETS = {'university': gen_arbitrary_network(ep_label=None, server_to_rack_channel_capacity=12500, num_channels=1, num_eps=64)}
# NETS = {'university': gen_fat_tree(k=4, N=30, num_channels=1)}
# NETS = {'university': gen_fat_tree(k=4, N=3, num_channels=1, rack_to_edge_channel_capacity=1250, edge_to_agg_channel_capacity=1250, agg_to_core_channel_capacity=1250)} # small network for quick benchmarking (10 Gbps == 1250 bytes/us)
NETS = {'university': gen_fat_tree(k=3, N=2, num_channels=1, server_to_rack_channel_capacity=500, rack_to_edge_channel_capacity=10000, edge_to_agg_channel_capacity=40000, agg_to_core_channel_capacity=40000)} # small network for quick benchmarking (10 Gbps == 1250 bytes/us)

# define network capacity for each benchmark
NETWORK_CAPACITIES = {'university': NETS['university'].graph['max_nw_capacity']}
NETWORK_EP_LINK_CAPACITIES = {'university': NETS['university'].graph['ep_link_capacity']}

# define network racks for each benchmark
RACKS_DICTS = {'university': NETS['university'].graph['rack_to_ep_dict']}








# # PRIVATE ENTERPRISE
# # -------------------------------------------------------------------------
# # define benchmarks to generate
# BENCHMARKS = ['private_enterprise']

# # define network topology for each benchmark
# # NETS = {'private_enterprise': gen_fat_tree(k=6, N=30, num_channels=1)}
# NETS = {'private_enterprise': gen_fat_tree(k=4, N=3, num_channels=1, rack_to_edge_channel_capacity=1250, edge_to_agg_channel_capacity=1250, agg_to_core_channel_capacity=1250)} # small network for quick benchmarking (10 Gbps == 1250 bytes/us)

# # define network capacity for each benchmark
# NETWORK_CAPACITIES = {'private_enterprise': NETS['private_enterprise'].graph['max_nw_capacity']}

# # define network racks for each benchmark
# RACKS_DICTS = {'private_enterprise': NETS['private_enterprise'].graph['rack_to_ep_dict']}







# # COMMERCIAL CLOUD
# # -------------------------------------------------------------------------
# # define benchmarks to generate
# BENCHMARKS = ['commercial_cloud']

# # define network topology for each benchmark
# # NETS = {'commercial_cloud': gen_fat_tree(k=11, N=30, num_channels=1)}
# NETS = {'commercial_cloud': gen_fat_tree(k=4, N=3, num_channels=1, rack_to_edge_channel_capacity=1250, edge_to_agg_channel_capacity=1250, agg_to_core_channel_capacity=1250)} # small network for quick benchmarking (10 Gbps == 1250 bytes/us)

# # define network capacity for each benchmark
# NETWORK_CAPACITIES = {'commercial_cloud': NETS['commercial_cloud'].graph['max_nw_capacity']}

# # define network racks for each benchmark
# RACKS_DICTS = {'commercial_cloud': NETS['commercial_cloud'].graph['rack_to_ep_dict']}






# # SOCIAL MEDIA CLOUD
# # -------------------------------------------------------------------------
# # define benchmarks to generate
# BENCHMARKS = ['social_media_cloud']

# # define network topology for each benchmark
# # NETS = {'social_media_cloud': gen_fat_tree(k=23, N=35, num_channels=1)}
# NETS = {'social_media_cloud': gen_fat_tree(k=4, N=3, num_channels=1, rack_to_edge_channel_capacity=1250, edge_to_agg_channel_capacity=1250, agg_to_core_channel_capacity=1250)} # small network for quick benchmarking (10 Gbps == 1250 bytes/us)

# # define network capacity for each benchmark
# NETWORK_CAPACITIES = {'social_media_cloud': NETS['social_media_cloud'].graph['max_nw_capacity']}

# # define network racks for each benchmark
# RACKS_DICTS = {'social_media_cloud': NETS['social_media_cloud'].graph['rack_to_ep_dict']}








# # UNIFORM 
# # -------------------------------------------------------------------------
# # define benchmarks to generate
# BENCHMARKS = ['uniform']

# # define network topology for each benchmark
# # NETS = {'uniform': gen_fat_tree(k=4, N=30, num_channels=1)}
# # NETS = {'uniform': gen_fat_tree(k=4, N=3, num_channels=1, rack_to_edge_channel_capacity=1250, edge_to_agg_channel_capacity=1250, agg_to_core_channel_capacity=1250)} # small network for quick benchmarking (10 Gbps == 1250 bytes/us)
# NETS = {'uniform': gen_fat_tree(k=3, N=2, num_channels=1, server_to_rack_channel_capacity=5, rack_to_edge_channel_capacity=5, edge_to_agg_channel_capacity=5, agg_to_core_channel_capacity=5)} # small network for quick benchmarking (10 Gbps == 1250 bytes/us)

# # define network capacity for each benchmark
# NETWORK_CAPACITIES = {'uniform': NETS['uniform'].graph['max_nw_capacity']}

# # define network racks for each benchmark
# RACKS_DICTS = {'uniform': None}









# # ARTIFICIAL LIGHT
# # -------------------------------------------------------------------------
# # define benchmarks to generate
# BENCHMARKS = ['artificial_light']

# # define network topology for each benchmark
# # NETS = {'university': gen_fat_tree(k=4, N=30, num_channels=1)}
# # NETS = {'university': gen_fat_tree(k=4, N=3, num_channels=1, rack_to_edge_channel_capacity=1250, edge_to_agg_channel_capacity=1250, agg_to_core_channel_capacity=1250)} # small network for quick benchmarking (10 Gbps == 1250 bytes/us)
# NETS = {'artificial_light': gen_fat_tree(k=3, N=2, num_channels=1, server_to_rack_channel_capacity=1, rack_to_edge_channel_capacity=500, edge_to_agg_channel_capacity=500, agg_to_core_channel_capacity=500)} # small network for quick benchmarking (10 Gbps == 1250 bytes/us)

# # define network capacity for each benchmark
# NETWORK_CAPACITIES = {'artificial_light': NETS['artificial_light'].graph['max_nw_capacity']}

# # define network racks for each benchmark
# RACKS_DICTS = {'artificial_light': NETS['artificial_light'].graph['rack_to_ep_dict']}




# -------------------------------------------------------------------------
# configuration validity check
# -------------------------------------------------------------------------
assert len(BENCHMARKS) == len(NETS.keys()) == len(NETWORK_CAPACITIES.keys()) == len(RACKS_DICTS.keys()), \
    'Must specify BENCHMARKS, NETS, NETWORK_CAPACITIES and RACKS_DICTS for each benchmark specified.'



