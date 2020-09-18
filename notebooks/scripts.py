import trafpy.generator as tpg
import config

import matplotlib.pyplot as plt



net = tpg.gen_fat_tree(k=6, N=30, num_channels=1)
fig = tpg.plot_network(net, draw_node_labels=False, network_node_size=1000, linewidths=1)


rack_prob_config = {'racks_dict': net.graph['rack_to_ep_dict'], 'prob_inter_rack': 0.7}
# rack_prob_config = None
# node_dist, _ = tpg.gen_uniform_node_dist(net.graph['endpoints'], rack_prob_config=rack_prob_config, show_fig=True, print_data=False)
node_dist, _ = tpg.gen_multimodal_node_dist(net.graph['endpoints'], rack_prob_config=rack_prob_config, show_fig=True, print_data=False)
