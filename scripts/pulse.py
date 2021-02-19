from trafpy.generator import Demand, DemandPlotter
import trafpy.generator as tpg
import time
import copy
from scipy.io import savemat


# set vars
N = 64
R = 3
path = r'C:/Users/benjo/Documents/Work/UCL/Workloads/'
endpoints = [str(i) for i in range(N)]
num_sets = 1 # init number of sets you want to generate
num_demands =  4800 # init number of demands you want to generate for each set
E = num_demands/(N*R)
L = 12
Lf = 0.9 # Load Factor
TD = 1
min_last_demand_arrival_time = 24000 # 24000 timeslot units if it’s a 2000 epoch simulation



# set any distributions you want to keep constant for all sets
flow_size_dist = tpg.gen_named_val_dist(dist='weibull',
                                        params={'_alpha': 4.8, '_lambda': 4.4},
                                        return_data=False,
                                        round_to_nearest=3)

interarrival_time_dist = tpg.gen_named_val_dist(dist='weibull',
                                               params={'_alpha': 3, '_lambda': 6},
                                               return_data=False,
                                               round_to_nearest=0.01)

node_dist = tpg.gen_multimodal_node_dist(eps=endpoints,
                                         skewed_nodes=[],
                                         skewed_node_probs=[],
                                         num_skewed_nodes=int(len(endpoints)*0.05))

# init network
load = 0.1
net = tpg.gen_arbitrary_network(ep_label=None, num_eps=N, server_to_rack_channel_capacity=250)
network_load_config = {'network_rate_capacity': net.graph['max_nw_capacity'], 
                       'ep_link_capacity': net.graph['ep_link_capacity'],
                       'target_load_fraction': load}

flow_centric_demand_data = tpg.create_demand_data(eps=endpoints,
                                                  node_dist=copy.deepcopy(node_dist),
                                                  flow_size_dist=flow_size_dist,
                                                  interarrival_time_dist=interarrival_time_dist,
                                                  network_load_config=network_load_config,
                                                  min_last_demand_arrival_time=None,
                                                  auto_node_dist_correction=False,
                                                  print_data=True)

demand = Demand(flow_centric_demand_data, net.graph['endpoints'])
plotter = DemandPlotter(demand)
plotter.plot_node_dist(eps=net.graph['endpoints'])
plotter.plot_node_load_dists(eps=net.graph['endpoints'], ep_link_bandwidth=net.graph['ep_link_capacity'])
