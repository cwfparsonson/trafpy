from trafpy.generator.src.networks import gen_fat_tree, gen_arbitrary_network

import numpy as np





# -------------------------------------------------------------------------
# general configuration
# -------------------------------------------------------------------------
# define benchmark version
BENCHMARK_VERSION = 'v001'

# define minimum number of demands to generate (may generate more to meet jensen_shannon_distance_threshold and/or min_last_demand_arrival_time)
MIN_NUM_DEMANDS = 10 # 1500 4000 10 400 10
MAX_NUM_DEMANDS = None # 2000 None 100 50

# define maximum allowed Jenson-Shannon distance for flow size and interarrival time distributions (lower value -> distributions must be more similar -> higher number of demands will be generated) (must be between 0 and 1)
JENSEN_SHANNON_DISTANCE_THRESHOLD = 0.1 # 0.1 0.2 0.5

# define minimum time of last demand's arrival (helps define minimum simulation time)
MIN_LAST_DEMAND_ARRIVAL_TIME = 3.2e5 # 3e3 300 units of us 3.2e5 2e6 3e6 6e7 6e8 None 3000.0 2e6 2.4e5pulse 2e6
# MIN_LAST_DEMAND_ARRIVAL_TIME = None

# define network load fractions
LOADS = np.arange(0.1, 1.0, 0.1).tolist()
# LOADS = [0.2, 0.8]
# LOADS = [0.1]
LOADS = [round(load, 3) for load in LOADS] # ensure no python floating point arithmetic errors

# define number of repetitions to perform for each benchmark for each load
NUM_REPEATS = 1

# define whether or not to auto correct invalid node distribution(s)
AUTO_NODE_DIST_CORRECTION = True

# slot size (if None, won't generate slots_dict database)
# SLOT_SIZE = None 
SLOT_SIZE = 1000.0 # 50.0 1000.0 10.0
# SLOT_SIZE = 10.0
# SLOT_SIZE = 10000.0










# -------------------------------------------------------------------------
# benchmark-specific configuration (uncomment one below)
# -------------------------------------------------------------------------

# FLOW-CENTRIC
# define benchmarks to generate
# BENCHMARKS = ['university']
# BENCHMARKS = ['private_enterprise']
# BENCHMARKS = ['commercial_cloud']
BENCHMARKS = ['social_media_cloud']

# BENCHMARKS = ['uniform']

# BENCHMARKS = ['skewed_nodes_sensitivity_0']
# BENCHMARKS = ['skewed_nodes_sensitivity_005']
# BENCHMARKS = ['skewed_nodes_sensitivity_01']
# BENCHMARKS = ['skewed_nodes_sensitivity_02']
# BENCHMARKS = ['skewed_nodes_sensitivity_04']

# BENCHMARKS = ['rack_sensitivity_0']
# BENCHMARKS = ['rack_sensitivity_02']
# BENCHMARKS = ['rack_sensitivity_04']
# BENCHMARKS = ['rack_sensitivity_06']
# BENCHMARKS = ['rack_sensitivity_08']


# JOB-CENTRIC
# BENCHMARKS = ['jobcentric_prototyping']
# BENCHMARKS = ['tensorflow']




# define network topology for each benchmark
net = gen_fat_tree(k=4, 
                   L=2, 
                   n=16, 
                   num_channels=1, 
                   server_to_rack_channel_capacity=1250, # 1250
                   rack_to_edge_channel_capacity=1000, 
                   edge_to_agg_channel_capacity=1000, 
                   agg_to_core_channel_capacity=2000)
NETS = {benchmark: net for benchmark in BENCHMARKS}

# define network capacity for each benchmark
NETWORK_CAPACITIES = {benchmark: net.graph['max_nw_capacity'] for benchmark in BENCHMARKS}
NETWORK_EP_LINK_CAPACITIES = {benchmark: net.graph['ep_link_capacity'] for benchmark in BENCHMARKS}

# define network racks for each benchmark
RACKS_DICTS = {benchmark: net.graph['rack_to_ep_dict'] for benchmark in BENCHMARKS}




