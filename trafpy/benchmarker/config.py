from trafpy.generator.src.networks import gen_fat_tree, gen_arbitrary_network

import numpy as np





# -------------------------------------------------------------------------
# general configuration
# -------------------------------------------------------------------------
# define benchmark version
BENCHMARK_VERSION = '0.0.1'

# define minimum number of demands to generate (may generate more to meet jensen_shannon_distance_threshold and/or min_last_demand_arrival_time)
MIN_NUM_DEMANDS = 10

# define maximum allowed Jenson-Shannon distance for flow size and interarrival time distributions (lower value -> distributions must be more similar -> higher number of demands will be generated) (must be between 0 and 1)
JENSEN_SHANNON_DISTANCE_THRESHOLD = 0.2 # 0.1

# define minimum time of last demand's arrival (helps define minimum simulation time)
MIN_LAST_DEMAND_ARRIVAL_TIME = 3.2e5 # units of us 2e6 3e6 6e7 6e8 None 3000.0 2e6 2.4e5pulse 2e6
# MIN_LAST_DEMAND_ARRIVAL_TIME = None

# define network load fractions
LOADS = np.arange(0.1, 1.0, 0.1).tolist()
# LOADS = [0.1]
LOADS = [round(load, 3) for load in LOADS] # ensure no python floating point arithmetic errors

# define number of repetitions to perform for each benchmark for each load
NUM_REPEATS = 1

# define whether or not to auto correct invalid node distribution(s)
AUTO_NODE_DIST_CORRECTION = True

# slot size (if None, won't generate slots_dict database)
# SLOT_SIZE = None 
SLOT_SIZE = 1000.0 # 50.0










# -------------------------------------------------------------------------
# benchmark-specific configuration (uncomment one below)
# -------------------------------------------------------------------------
# define benchmarks to generate
# BENCHMARKS = ['university']
# BENCHMARKS = ['private_enterprise']
# BENCHMARKS = ['commercial_cloud']
# BENCHMARKS = ['social_media_cloud']

# BENCHMARKS = ['uniform']
# BENCHMARKS = ['artificial_light']

# BENCHMARKS = ['skewed_nodes_sensitivity_0']
BENCHMARKS = ['skewed_nodes_sensitivity_0.05']
# BENCHMARKS = ['skewed_nodes_sensitivity_0.1']
# BENCHMARKS = ['skewed_nodes_sensitivity_0.2']
# BENCHMARKS = ['skewed_nodes_sensitivity_0.4']

# BENCHMARKS = ['rack_dist_sensitivity_0']
# BENCHMARKS = ['rack_dist_sensitivity_0.2']
# BENCHMARKS = ['rack_dist_sensitivity_0.4']
# BENCHMARKS = ['rack_dist_sensitivity_0.6']
# BENCHMARKS = ['rack_dist_sensitivity_0.8']

# define network topology for each benchmark
net = gen_fat_tree(k=4, 
                   L=2, 
                   n=16, 
                   num_channels=1, 
                   server_to_rack_channel_capacity=1250, 
                   rack_to_edge_channel_capacity=1000, 
                   edge_to_agg_channel_capacity=1000, 
                   agg_to_core_channel_capacity=2000, 
                   bidirectional_links=True)
NETS = {benchmark: net for benchmark in BENCHMARKS}

# define network capacity for each benchmark
NETWORK_CAPACITIES = {benchmark: net.graph['max_nw_capacity'] for benchmark in BENCHMARKS}
NETWORK_EP_LINK_CAPACITIES = {benchmark: net.graph['ep_link_capacity'] for benchmark in BENCHMARKS}

# define network racks for each benchmark
RACKS_DICTS = {benchmark: net.graph['rack_to_ep_dict'] for benchmark in BENCHMARKS}







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








# # UNIVERSITY
# # -------------------------------------------------------------------------
# # define benchmarks to generate
# BENCHMARKS = ['university']

# # define network topology for each benchmark
# NETS = {'university': gen_fat_tree(k=10, 
                                   # L=2, 
                                   # n=16, 
                                   # num_channels=1, 
                                   # server_to_rack_channel_capacity=1250, 
                                   # rack_to_edge_channel_capacity=1000, 
                                   # edge_to_agg_channel_capacity=1000, 
                                   # agg_to_core_channel_capacity=2000, 
                                   # bidirectional_links=True)} # small network for quick benchmarking (10 Gbps == 1250 bytes/us)

# # define network capacity for each benchmark
# NETWORK_CAPACITIES = {'university': NETS['university'].graph['max_nw_capacity']}
# NETWORK_EP_LINK_CAPACITIES = {'university': NETS['university'].graph['ep_link_capacity']}

# # define network racks for each benchmark
# RACKS_DICTS = {'university': NETS['university'].graph['rack_to_ep_dict']}
# # RACKS_DICTS = {'university': None} # PULSE








# # PRIVATE ENTERPRISE
# # -------------------------------------------------------------------------
# # define benchmarks to generate
# BENCHMARKS = ['private_enterprise']

# # define network topology for each benchmark
# # NETS = {'private_enterprise': gen_fat_tree(k=6, N=30, num_channels=1)}
# # NETS = {'private_enterprise': gen_fat_tree(k=4, N=3, num_channels=1, rack_to_edge_channel_capacity=1250, edge_to_agg_channel_capacity=1250, agg_to_core_channel_capacity=1250)} # small network for quick benchmarking (10 Gbps == 1250 bytes/us)
# # NETS = {'private_enterprise': gen_fat_tree(k=3, N=2, num_channels=1, server_to_rack_channel_capacity=500, rack_to_edge_channel_capacity=10000, edge_to_agg_channel_capacity=40000, agg_to_core_channel_capacity=40000, bidirectional_links=True)} # small network for quick benchmarking (10 Gbps == 1250 bytes/us)
# NETS = {'private_enterprise': gen_fat_tree(k=4, L=2, n=4, num_channels=1, server_to_rack_channel_capacity=500, rack_to_edge_channel_capacity=1000, edge_to_agg_channel_capacity=1000, agg_to_core_channel_capacity=2000, bidirectional_links=True)} # small network for quick benchmarking (10 Gbps == 1250 bytes/us)

# # define network capacity for each benchmark
# NETWORK_CAPACITIES = {'private_enterprise': NETS['private_enterprise'].graph['max_nw_capacity']} 
# NETWORK_EP_LINK_CAPACITIES = {'private_enterprise': NETS['private_enterprise'].graph['ep_link_capacity']}

# # define network racks for each benchmark
# RACKS_DICTS = {'private_enterprise': NETS['private_enterprise'].graph['rack_to_ep_dict']}







# # COMMERCIAL CLOUD
# # -------------------------------------------------------------------------
# # define benchmarks to generate
# BENCHMARKS = ['commercial_cloud']

# # define network topology for each benchmark
# # NETS = {'commercial_cloud': gen_fat_tree(k=11, N=30, num_channels=1)}
# # NETS = {'commercial_cloud': gen_fat_tree(k=4, N=3, num_channels=1, rack_to_edge_channel_capacity=1250, edge_to_agg_channel_capacity=1250, agg_to_core_channel_capacity=1250)} # small network for quick benchmarking (10 Gbps == 1250 bytes/us)
# # NETS = {'commercial_cloud': gen_fat_tree(k=3, N=2, num_channels=1, server_to_rack_channel_capacity=500, rack_to_edge_channel_capacity=10000, edge_to_agg_channel_capacity=40000, agg_to_core_channel_capacity=40000, bidirectional_links=True)} # small network for quick benchmarking (10 Gbps == 1250 bytes/us)
# # NETS = {'commercial_cloud': gen_fat_tree(k=4, N=2, num_channels=1, server_to_rack_channel_capacity=500, rack_to_edge_channel_capacity=10000, edge_to_agg_channel_capacity=40000, agg_to_core_channel_capacity=40000, bidirectional_links=True)} # small network for quick benchmarking (10 Gbps == 1250 bytes/us)
# NETS = {'commercial_cloud': gen_fat_tree(k=4, L=2, n=4, num_channels=1, server_to_rack_channel_capacity=500, rack_to_edge_channel_capacity=1000, edge_to_agg_channel_capacity=1000, agg_to_core_channel_capacity=2000, bidirectional_links=True)} # small network for quick benchmarking (10 Gbps == 1250 bytes/us)

# # define network capacity for each benchmark
# NETWORK_CAPACITIES = {'commercial_cloud': NETS['commercial_cloud'].graph['max_nw_capacity']}
# NETWORK_EP_LINK_CAPACITIES = {'commercial_cloud': NETS['commercial_cloud'].graph['ep_link_capacity']}

# # define network racks for each benchmark
# RACKS_DICTS = {'commercial_cloud': NETS['commercial_cloud'].graph['rack_to_ep_dict']}






# # SOCIAL MEDIA CLOUD
# # -------------------------------------------------------------------------
# # define benchmarks to generate
# BENCHMARKS = ['social_media_cloud']

# # define network topology for each benchmark
# # NETS = {'social_media_cloud': gen_fat_tree(k=23, N=35, num_channels=1)}
# # NETS = {'social_media_cloud': gen_fat_tree(k=4, N=3, num_channels=1, rack_to_edge_channel_capacity=1250, edge_to_agg_channel_capacity=1250, agg_to_core_channel_capacity=1250)} # small network for quick benchmarking (10 Gbps == 1250 bytes/us)
# # NETS = {'social_media_cloud': gen_fat_tree(k=3, N=2, num_channels=1, server_to_rack_channel_capacity=500, rack_to_edge_channel_capacity=10000, edge_to_agg_channel_capacity=40000, agg_to_core_channel_capacity=40000, bidirectional_links=True)} # small network for quick benchmarking (10 Gbps == 1250 bytes/us)
# # NETS = {'social_media_cloud': gen_fat_tree(k=4, N=2, num_channels=1, server_to_rack_channel_capacity=500, rack_to_edge_channel_capacity=10000, edge_to_agg_channel_capacity=40000, agg_to_core_channel_capacity=40000, bidirectional_links=True)} # small network for quick benchmarking (10 Gbps == 1250 bytes/us)
# NETS = {'social_media_cloud': gen_fat_tree(k=4, L=2, n=4, num_channels=1, server_to_rack_channel_capacity=500, rack_to_edge_channel_capacity=1000, edge_to_agg_channel_capacity=1000, agg_to_core_channel_capacity=2000, bidirectional_links=True)} # small network for quick benchmarking (10 Gbps == 1250 bytes/us)

# # define network capacity for each benchmark
# NETWORK_CAPACITIES = {'social_media_cloud': NETS['social_media_cloud'].graph['max_nw_capacity']}
# NETWORK_EP_LINK_CAPACITIES = {'social_media_cloud': NETS['social_media_cloud'].graph['ep_link_capacity']}

# # define network racks for each benchmark
# RACKS_DICTS = {'social_media_cloud': NETS['social_media_cloud'].graph['rack_to_ep_dict']}







# SKEWED NODES SENSITIVITY

# -------------------------------------------------------------------------
# # define benchmarks to generate
# # BENCHMARKS = ['skewed_nodes_sensitivity_0']
# # BENCHMARKS = ['skewed_nodes_sensitivity_0.05']
# # BENCHMARKS = ['skewed_nodes_sensitivity_0.1']
# # BENCHMARKS = ['skewed_nodes_sensitivity_0.2']
# BENCHMARKS = ['skewed_nodes_sensitivity_0.4']

# # define network topology for each benchmark
# net = gen_fat_tree(k=4, 
                   # L=2, 
                   # n=16, 
                   # num_channels=1, 
                   # server_to_rack_channel_capacity=1250, 
                   # rack_to_edge_channel_capacity=1000, 
                   # edge_to_agg_channel_capacity=1000, 
                   # agg_to_core_channel_capacity=2000, 
                   # bidirectional_links=True)
# NETS = {benchmark: net for benchmark in BENCHMARKS}

# # define network capacity for each benchmark
# NETWORK_CAPACITIES = {benchmark: net.graph['max_nw_capacity'] for benchmark in BENCHMARKS}
# NETWORK_EP_LINK_CAPACITIES = {benchmark: net.graph['ep_link_capacity'] for benchmark in BENCHMARKS}

# # define network racks for each benchmark
# RACKS_DICTS = {benchmark: net.graph['rack_to_ep_dict'] for benchmark in BENCHMARKS}


















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
# NETS = {'artificial_light': gen_fat_tree(k=2, N=2, num_channels=1, server_to_rack_channel_capacity=10, rack_to_edge_channel_capacity=500, edge_to_agg_channel_capacity=500, agg_to_core_channel_capacity=500)} small network for quick benchmarking (10 Gbps == 1250 bytes/us)

# # define network capacity for each benchmark
# NETWORK_CAPACITIES = {'artificial_light': NETS['artificial_light'].graph['max_nw_capacity']}
# NETWORK_EP_LINK_CAPACITIES = {'artificial_light': NETS['artificial_light'].graph['ep_link_capacity']}

# # define network racks for each benchmark
# RACKS_DICTS = {'artificial_light': None}














# -------------------------------------------------------------------------
# configuration validity check
# -------------------------------------------------------------------------
# assert len(BENCHMARKS) == len(NETS.keys()) == len(NETWORK_CAPACITIES.keys()) == len(RACKS_DICTS.keys()), \
    # 'Must specify BENCHMARKS, NETS, NETWORK_CAPACITIES and RACKS_DICTS for each benchmark specified.'



