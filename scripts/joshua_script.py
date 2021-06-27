from trafpy.benchmarker import BenchmarkImporter
from trafpy.generator.src.tools import save_data_as_json, save_data_as_csv, pickle_data
import trafpy.generator as tpg
import time
import copy
from scipy.io import savemat


# set vars
N = 64
path = r'C:/Users/benjo/Documents/Work/UCL/Workloads/'
min_num_demands = 200
min_last_demand_arrival_time = 10000
jensen_shannon_distance_threshold = 0.1
# loads = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
loads = [0.9, 0.8]

# init network
net = tpg.gen_arbitrary_network(ep_label=None, num_eps=N, server_to_rack_channel_capacity=50000, bidirectional_links=True)

# set flow size, interarrival time, and node distributions
importer = BenchmarkImporter(benchmark_version='v001', load_prev_dists=False)
benchmark_dists = importer.get_benchmark_dists(benchmark='university', racks_dict=None, eps=net.graph['endpoints'])
flow_size_dist, interarrival_time_dist, node_dist = benchmark_dists['flow_size_dist'], benchmark_dists['interarrival_time_dist'], benchmark_dists['node_dist']

# plot dists
tpg.plot_dict_scatter(_dict=interarrival_time_dist, marker_size=30, logscale=False, rand_var_name='Interarrival Time (us)')
tpg.plot_dict_scatter(_dict=flow_size_dist, marker_size=30, logscale=True, rand_var_name='Flow Size (Bytes)')
tpg.plot_node_dist(node_dist, chord_edge_width_range=[1,25], chord_edge_display_threshold=0.475)

from trafpy.generator import Demand, DemandPlotter

demands = []
for load in loads:
    print('Generating load {}...'.format(load))
    
    start = time.time()
    
    # generate demands
    network_load_config = {'network_rate_capacity': net.graph['max_nw_capacity'], 
                           'ep_link_capacity': net.graph['ep_link_capacity'],
                           'target_load_fraction': load}
    flow_centric_demand_data = tpg.create_demand_data(eps=net.graph['endpoints'],
                                                      node_dist=node_dist,
                                                      flow_size_dist=flow_size_dist,
                                                      interarrival_time_dist=interarrival_time_dist,
                                                      network_load_config=network_load_config,
                                                      jensen_shannon_distance_threshold=jensen_shannon_distance_threshold,
                                                      min_num_demands=min_num_demands,
                                                      min_last_demand_arrival_time=min_last_demand_arrival_time,
                                                      check_dont_exceed_one_ep_load=True,
                                                      auto_node_dist_correction=True,
                                                      bidirectional_links=True,
                                                      print_data=True)
    save_data_as_json(path_to_save='data2/joshua_pulse_load_{}'.format(load), data=flow_centric_demand_data, overwrite=False)
    
    # # plot generated demands
    # demand = Demand(flow_centric_demand_data, net.graph['endpoints'])
    # demands.append(demand)
    # plotter = DemandPlotter(demand)
# #     plotter.plot_flow_size_dist(logscale=True, num_bins=30)
# #     plotter.plot_interarrival_time_dist(logscale=False, num_bins=30)
    # plotter.plot_node_dist(eps=net.graph['endpoints'], chord_edge_width_range=[1,25], chord_edge_display_threshold=0.2)
    # plotter.plot_node_load_dists(eps=net.graph['endpoints'], ep_link_bandwidth=net.graph['ep_link_capacity'], plot_extras=False)
    
    # # save generated demands
    # savemat("benchmark/University/load{}_matlab_matrix.mat".format(load), flow_centric_demand_data)
    
    end = time.time()
    print('Generated load {} in {} seconds.'.format(load, end-start))
