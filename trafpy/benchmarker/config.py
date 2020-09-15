from trafpy.generator.src.networks import gen_fat_tree

NET = gen_fat_tree(k=4, N=30, num_channels=1)
NETWORK_CAPACITY = NET.graph['max_nw_capacity']
RACKS_DICT = NET.graph['rack_to_ep_dict']

