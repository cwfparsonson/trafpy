import trafpy.generator as tpg
import config

import matplotlib.pyplot as plt
import json
import time



net = tpg.gen_fat_tree(k=6, N=30, num_channels=1)
fig = tpg.plot_network(net, draw_node_labels=False, network_node_size=1000, linewidths=1)

num_skewed_nodes = int(0.2 * len(net.graph['endpoints']))
skewed_node_probs = [0.8/num_skewed_nodes for _ in range(num_skewed_nodes)]
figs = []
for prob_inter_rack in [0.5]:
    rack_prob_config = {'racks_dict': net.graph['rack_to_ep_dict'], 'prob_inter_rack': prob_inter_rack}
    start = time.time()
    node_dist, fig = tpg.gen_multimodal_node_dist(net.graph['endpoints'], rack_prob_config=rack_prob_config, num_skewed_nodes=num_skewed_nodes, skewed_node_probs=skewed_node_probs, plot_fig=True, show_fig=False, print_data=False)
    figs.append(fig)
    end = time.time()
    print('Generated in {} s'.format(end-start))
for fig in figs:
    plt.show()



# rack_prob_config = {'racks_dict': net.graph['rack_to_ep_dict'], 'prob_inter_rack': 0.5}
# rack_prob_config = None
# node_dist, _ = tpg.gen_uniform_node_dist(net.graph['endpoints'], rack_prob_config=rack_prob_config, show_fig=True, print_data=False)

# node_dist, _ = tpg.gen_multimodal_node_dist(net.graph['endpoints'], rack_prob_config=rack_prob_config, show_fig=True, print_data=False)
# node_dist, _ = tpg.gen_multimodal_node_pair_dist(net.graph['endpoints'], num_skewed_pairs=12500, rack_prob_config=rack_prob_config, show_fig=True, print_data=False)


# pair_prob_dict = tpg.get_pair_prob_dict_of_node_dist_matrix(node_dist, net.graph['endpoints'])
# fig = tpg.plot_val_bar(x_values=list(pair_prob_dict.keys()),
                       # y_values=list(pair_prob_dict.values()),
                       # ylabel='Probability',
                       # xlabel='Node Pair',
                       # plot_x_ticks=False,
                       # show_fig=False)
# plt.savefig('/home/cwfparsonson/Downloads/node_pair_bar_chart')


# # plot bar for all inter-rack pairs
# ep_to_rack = net.graph['ep_to_rack_dict']
# intra_rack_pair_prob_dict = {}
# inter_rack_pair_prob_dict = {}
# for pair in list(pair_prob_dict.keys()):
    # pair_loaded = json.loads(pair)
    # src, dst = pair_loaded[0], pair_loaded[1]
    # if ep_to_rack[src] == ep_to_rack[dst]:
        # # intra-rack
        # try:
            # intra_rack_pair_prob_dict[pair] = pair_prob_dict[pair]
        # except KeyError:
            # pair = json.loads(pair)
            # pair = json.dumps([pair[1],pair[0]])
            # intra_rack_pair_prob_dict[pair] = pair_prob_dict[pair]
    # else:
        # # inter-rack
        # try:
            # inter_rack_pair_prob_dict[pair] = pair_prob_dict[pair]
        # except KeyError:
            # pair = json.loads(pair)
            # pair = json.dumps([pair[1],pair[0]])
            # inter_rack_pair_prob_dict[pair] = pair_prob_dict[pair]

# fig = tpg.plot_val_bar(x_values=list(intra_rack_pair_prob_dict.keys()),
                       # y_values=list(intra_rack_pair_prob_dict.values()),
                       # ylabel='Probability',
                       # xlabel='Intra-Rack Node Pair',
                       # plot_x_ticks=False,
                       # show_fig=False)
# plt.savefig('/home/cwfparsonson/Downloads/intra_rack_node_pair_bar_chart')


# fig = tpg.plot_val_bar(x_values=list(inter_rack_pair_prob_dict.keys()),
                       # y_values=list(inter_rack_pair_prob_dict.values()),
                       # ylabel='Probability',
                       # xlabel='Inter-Rack Node Pair',
                       # plot_x_ticks=False,
                       # show_fig=False)
# plt.savefig('/home/cwfparsonson/Downloads/inter_rack_node_pair_bar_chart')







