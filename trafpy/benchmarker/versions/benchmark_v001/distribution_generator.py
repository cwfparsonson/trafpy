import trafpy
from trafpy.benchmarker.versions.benchmark_v001 import config
from trafpy.generator.src.dists import node_dists
from trafpy.generator.src.dists import val_dists
from trafpy.generator.src.dists import plot_dists
from trafpy.generator.src.tools import load_data_from_json, save_data_as_json
import os
import json

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import cv2
import io
import copy




class DistributionGenerator:
    def __init__(self, load_prev_dists=True):
        self.load_prev_dists=load_prev_dists
        self.benchmark_version = '0.0.1'
        self.valid_benchmark_sets = config.ALL_BENCHMARK_SETS

        trafpy_path = os.path.dirname(trafpy.__file__)
        self.benchmark_version_path = trafpy_path + '/benchmarker/versions/benchmark_v001/'

        self.dist_names = ['node_dist', 'interarrival_time_dist', 'flow_size_dist']



    def init_dir(self, benchmark):
        # check if data folder exists
        if os.path.exists(self.benchmark_version_path+'data'):
            # data folder already exists
            pass
        else:
            print('Creating data folder in {}'.format(self.benchmark_version_path))
            os.mkdir(self.benchmark_version_path+'data')

        # check if benchmark folder exists in data folder
        if os.path.exists(self.benchmark_version_path+'data/'+str(benchmark)):
            # benchmark folder already exists
            pass
        else:
            print('Creating {} benchmark folder in {}'.format(benchmark, self.benchmark_version_path+'data/'))
            os.mkdir(self.benchmark_version_path+'data/'+benchmark)

    def load_dist(self, benchmark, dist_name):
        # check if dists already previously saved 
        path_to_data = self.benchmark_version_path+'data/{}/{}.json'.format(benchmark, dist_name)
        if os.path.exists(path_to_data):
            print('Loading previously saved benchmark dists from {}'.format(path_to_data))
            dist = load_data_from_json(path_to_load=path_to_data, print_times=True)
        else:
            print('{} distribution not previously saved in {}.'.format(dist_name, path_to_data))
            dist = None

        return dist, path_to_data


    def get_node_dist(self, benchmark, racks_dict, eps, save_data=True):
        self.init_dir(benchmark)

        if benchmark not in self.valid_benchmark_sets:
            raise Exception('Unrecognised benchmark set \'{}\'. Valid benchmark sets for benchmark {}:\n{}'.format(benchmark, self.benchmark_version, self.valid_benchmark_sets))


        num_skewed_nodes = int(0.2 * len(eps))
        skewed_node_probs = [0.55/num_skewed_nodes for _ in range(num_skewed_nodes)]

        dist_name = 'node_dist'
        node_dist = None
        if self.load_prev_dists:
            # attempt to load previously saved dist
            node_dist, path_to_data = self.load_dist(benchmark, dist_name=dist_name)
        else:
            # just get path_to_data for saving
            path_to_data = self.benchmark_version_path+'data/{}/{}.json'.format(benchmark, dist_name)

        if node_dist is None or not self.load_prev_dists:
            print('Generating {} distribution for {} benchmark...'.format(dist_name, benchmark))

            if benchmark == 'university':
                rack_prob_config = {'racks_dict': racks_dict, 'prob_inter_rack': 0.7}
                # node_dist = node_dists.gen_uniform_node_dist(eps, rack_prob_config=rack_prob_config, show_fig=False, print_data=False)
                node_dist = node_dists.gen_multimodal_node_dist(eps, rack_prob_config=rack_prob_config, num_skewed_nodes=num_skewed_nodes, skewed_node_probs=skewed_node_probs, show_fig=False, print_data=False)
            
            elif benchmark == 'private_enterprise':
                rack_prob_config = {'racks_dict': racks_dict, 'prob_inter_rack': 0.5}
                # node_dist = node_dists.gen_uniform_node_dist(eps, rack_prob_config=rack_prob_config, show_fig=False, print_data=False)
                node_dist = node_dists.gen_multimodal_node_dist(eps, rack_prob_config=rack_prob_config, num_skewed_nodes=num_skewed_nodes, skewed_node_probs=skewed_node_probs, show_fig=False, print_data=False)

            elif benchmark == 'commercial_cloud':
                rack_prob_config = {'racks_dict': racks_dict, 'prob_inter_rack': 0.2}
                # node_dist = node_dists.gen_uniform_node_dist(eps, rack_prob_config=rack_prob_config, show_fig=False, print_data=False)
                node_dist = node_dists.gen_multimodal_node_dist(eps, rack_prob_config=rack_prob_config, num_skewed_nodes=num_skewed_nodes, skewed_node_probs=skewed_node_probs, show_fig=False, print_data=False)

            elif benchmark == 'social_media_cloud':
                rack_prob_config = {'racks_dict': racks_dict, 'prob_inter_rack': 0.129}
                # node_dist = node_dists.gen_uniform_node_dist(eps, rack_prob_config=rack_prob_config, show_fig=False, print_data=False)
                node_dist = node_dists.gen_multimodal_node_dist(eps, rack_prob_config=rack_prob_config, num_skewed_nodes=num_skewed_nodes, skewed_node_probs=skewed_node_probs, show_fig=False, print_data=False)

            else:
                raise Exception('Benchmark \'{}\' not recognised.'.format(benchmark))


            if save_data:
                save_data_as_json(path_to_save=path_to_data, data=node_dist, overwrite=True, print_times=False)


        return node_dist


    def get_flow_size_dist(self, benchmark, save_data=True):
        self.init_dir(benchmark)

        if benchmark not in self.valid_benchmark_sets:
            raise Exception('Unrecognised benchmark set \'{}\'. Valid benchmark sets for benchmark {}:\n{}'.format(benchmark, self.benchmark_version, self.valid_benchmark_sets))

        dist_name = 'flow_size_dist'
        flow_size_dist = None
        if self.load_prev_dists:
            # attempt to load previously saved dist
            flow_size_dist, path_to_data = self.load_dist(benchmark, dist_name=dist_name)
        else:
            # just get path_to_data for saving
            path_to_data = self.benchmark_version_path+'data/{}/{}.json'.format(benchmark, dist_name)

        if flow_size_dist is None or not self.load_prev_dists:
            print('Generating {} distribution for {} benchmark...'.format(dist_name, benchmark))

            if benchmark == 'university':
                flow_size_dist = val_dists.gen_named_val_dist(dist='lognormal',
                                                              params={'_mu': 7, '_sigma': 2.5},
                                                              min_val=1,
                                                              show_fig=False,
                                                              print_data=False,
                                                              round_to_nearest=1)

            elif benchmark == 'private_enterprise':
                flow_size_dist = val_dists.gen_named_val_dist(dist='lognormal',
                                                              params={'_mu': 7, '_sigma': 2.5},
                                                              min_val=1,
                                                              show_fig=False,
                                                              print_data=False,
                                                              round_to_nearest=1)

            elif benchmark == 'commercial_cloud':
                flow_size_dist = val_dists.gen_named_val_dist(dist='lognormal',
                                                              params={'_mu': 7, '_sigma': 2.5},
                                                              min_val=1,
                                                              show_fig=False,
                                                              print_data=False,
                                                              round_to_nearest=1)

            elif benchmark == 'social_media_cloud':
                flow_size_dist = val_dists.gen_named_val_dist(dist='weibull',
                                                              params={'_alpha': 0.5, '_lambda': 21000},
                                                              min_val=1,
                                                              show_fig=False,
                                                              print_data=False,
                                                              round_to_nearest=1)

            else:
                raise Exception('Benchmark \'{}\' not recognised.'.format(benchmark))


            if save_data:
                save_data_as_json(path_to_save=path_to_data, data=flow_size_dist, overwrite=True, print_times=False)



        return flow_size_dist




    def get_interarrival_time_dist(self, benchmark, save_data=True):
        self.init_dir(benchmark)

        if benchmark not in self.valid_benchmark_sets:
            raise Exception('Unrecognised benchmark set \'{}\'. Valid benchmark sets for benchmark {}:\n{}'.format(benchmark, self.benchmark_version, self.valid_benchmark_sets))

        dist_name = 'interarrival_time_dist'
        interarrival_time_dist = None
        if self.load_prev_dists:
            # attempt to load previously saved dist
            interarrival_time_dist, path_to_data = self.load_dist(benchmark, dist_name=dist_name)
        else:
            # just get path_to_data for saving
            path_to_data = self.benchmark_version_path+'data/{}/{}.json'.format(benchmark, dist_name)

        if interarrival_time_dist is None or not self.load_prev_dists:
            print('Generating {} distribution for {} benchmark...'.format(dist_name, benchmark))

            if benchmark == 'university':
                interarrival_time_dist = val_dists.gen_named_val_dist(dist='weibull',
                                                                      params={'_alpha': 0.9, '_lambda': 6000},
                                                                      show_fig=False,
                                                                      print_data=False,
                                                                      round_to_nearest=1)

            elif benchmark == 'private_enterprise':
                interarrival_time_dist = val_dists.gen_multimodal_val_dist(min_val=1,
                                                                           max_val=100000,
                                                                           locations=[40,1],
                                                                           skews=[-1,4],
                                                                           scales=[60,1000],
                                                                           num_skew_samples=[10000,10000],
                                                                           bg_factor=0.05)

            elif benchmark == 'commercial_cloud':
                interarrival_time_dist = val_dists.gen_multimodal_val_dist(min_val=1,
                                                                           max_val=10000,
                                                                           locations=[10,20,100,1],
                                                                           skews=[0,0,0,100],
                                                                           scales=[1,3,4,50],
                                                                           num_skew_samples=[10000,7000,5000,20000],
                                                                           bg_factor=0.01)

            elif benchmark == 'social_media_cloud':
                interarrival_time_dist = val_dists.gen_named_val_dist(dist='lognormal',
                                                                      params={'_mu': 6, '_sigma': 2.3},
                                                                      show_fig=False,
                                                                      print_data=False,
                                                                      round_to_nearest=1)

            else:
                raise Exception('Benchmark \'{}\' not recognised.'.format(benchmark))


            if save_data:
                save_data_as_json(path_to_save=path_to_data, data=interarrival_time_dist, overwrite=True, print_times=False)


        return interarrival_time_dist
    


    def plot_benchmark_dists(self):
        '''Plots dist info of all benchmarks.'''

        num_demands = 1000

        # load dists
        dists = {benchmark: {dist_name: None for dist_name in self.dist_names} for benchmark in self.valid_benchmark_sets}
        plotted_rand_vars = copy.deepcopy(dists)
        plots = copy.deepcopy(dists)
        for benchmark in self.valid_benchmark_sets:
            for dist_name in self.dist_names:
                # store dist
                dists[benchmark][dist_name], path_to_data = self.load_dist(benchmark, dist_name)
                dists[benchmark][dist_name] = json.loads(dists[benchmark][dist_name])

                # check loaded dist successfully
                if dists[benchmark][dist_name] is None:
                    raise Exception('Dist {} for benchmark {} not found in {}. Ensure dist is named as one of {}, and that dist has been saved in correct location.'.format(dist_name, benchmark, path_to_data, self.dist_names))


                if dist_name == 'flow_size_dist' or dist_name == 'interarrival_time_dist':
                    # remove str keys
                    dists[benchmark][dist_name] = {float(key): val for key, val in dists[benchmark][dist_name].items()}

                    # generate random variables from dist to plot
                    rand_vars = val_dists.gen_rand_vars_from_discretised_dist(unique_vars=list(dists[benchmark][dist_name].keys()),
                                                                                probabilities=list(dists[benchmark][dist_name].values()),
                                                                                num_demands=num_demands)
                    plotted_rand_vars[benchmark][dist_name] = rand_vars

                    if dist_name == 'flow_size_dist':
                        # plot flow size dist
                        fig = plot_dists.plot_val_dist(rand_vars, show_fig=False, plot_horizontally=False, logscale=True, num_bins=20, rand_var_name='Flow Size (Bytes)')
                        plots[benchmark][dist_name] = fig
                    elif dist_name == 'interarrival_time_dist':
                        fig = plot_dists.plot_val_dist(rand_vars, show_fig=False, plot_horizontally=False, logscale=True, num_bins=20, rand_var_name='Interarrival Time (us)')
                        plots[benchmark][dist_name] = fig

                elif dist_name == 'node_dist':
                    fig = plot_dists.plot_node_dist(dists[benchmark][dist_name]) 
                    plots[benchmark][dist_name] = fig

                else:
                    raise Exception('Unrecognised dist_name {}'.format(dist_name))
                    
        
        # create plot of sub plots
        fig = plt.figure()
        ax = fig.gca()
        ax.axis('off')
        ax.margins(0) # remove large white borders
        fig.tight_layout(pad=0)
        fig.subplots_adjust(hspace=0.1, wspace=0.1)
        i = 1
        # for benchmark in self.valid_benchmark_sets:
            # for dist_name in self.dist_names:
        for dist_name in self.dist_names:
            for benchmark in self.valid_benchmark_sets:
                # fig.add_subplot(len(self.valid_benchmark_sets), len(self.dist_names), i)
                fig.add_subplot(len(self.dist_names), len(self.valid_benchmark_sets), i)
                ax = fig.gca()
                ax.axis('off')
                dist_fig = plots[benchmark][dist_name]
                img = self.conv_fig_to_image(dist_fig)
                # img = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
                # img = np.frombuffer(dist_fig.canvas.tostring_rgb(), dtype=np.uint8)
                # img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                plt.imshow(img)
                i += 1
                # axs[benchmark_to_idx[benchmark], dist_to_idx[dist_name]] = plots[benchmark][dist_name]


        plt.show()

        
        return plots, dists, plotted_rand_vars



                    
    def conv_fig_to_image(self, fig, dpi=300):
        '''
        Takes matplotlib figure and converts into numpy array of RGB pixel values
        '''
        # minimise whitespace around edges
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        buf = io.BytesIO()
        try:
            fig.savefig(buf, bbox_inches='tight', pad_inches=0, format='png', dpi=dpi)
        except AttributeError:
            fig.figure.savefig(buf, bbox_inches='tight', pad_inches=0, format='png', dpi=dpi)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr,1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img









        






