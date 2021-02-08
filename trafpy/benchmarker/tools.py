from trafpy.benchmarker import config
from trafpy.generator.src.builder import create_demand_data
from trafpy.generator.src.tools import save_data_as_json, save_data_as_csv
from trafpy.benchmarker.versions.benchmark_importer import BenchmarkImporter

import numpy as np
import time





def gen_benchmark_demands(path_to_save=None, save_format='json', load_prev_dists=True):
    # init benchmark importer
    importer = BenchmarkImporter(config.BENCHMARK_VERSION, load_prev_dists=load_prev_dists)

    # load distributions for each benchmark
    benchmark_dists = {benchmark: {} for benchmark in config.BENCHMARKS}
    benchmark_demands = {benchmark: {load: {repeat: {} for repeat in range(config.NUM_REPEATS)} for load in config.LOADS} for benchmark in config.BENCHMARKS}

    # begin generating data for each benchmark
    num_loads = len(config.LOADS)
    start_loops = time.time()
    print('\n~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*')
    print('Benchmarks to Generate: {}'.format(config.BENCHMARKS))
    print('Loads to generate: {}'.format(config.LOADS))
    print('Number of repeats to generate for each benchmark load: {}'.format(config.NUM_REPEATS))
    for benchmark in config.BENCHMARKS:
        print('~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*')
        print('Generating demands for benchmark \'{}\'...'.format(benchmark))
        
        # get racks and endpoints
        racks_dict = config.RACKS_DICTS[benchmark]
        if racks_dict is not None:
            eps_racks_list = [eps for eps in racks_dict.values()]
            eps = []
            for rack in eps_racks_list:
                for ep in rack:
                    eps.append(ep)
        else:
            eps = config.NETS[benchmark].graph['endpoints']

        start_benchmark = time.time()
        load_counter = 1
        benchmark_dists[benchmark] = importer.get_benchmark_dists(benchmark, racks_dict, eps)
        for load in config.LOADS:
            start_load = time.time()
            network_load_config = {'network_rate_capacity': config.NETWORK_CAPACITIES[benchmark], 
                                   'ep_link_capacity': config.NETWORK_EP_LINK_CAPACITIES[benchmark],
                                   'target_load_fraction': load,
                                   'disable_timeouts': True}
            # print('\n~~~~~~ network load config ~~~~~~~\n{}'.format(network_load_config))
            for repeat in range(config.NUM_REPEATS):
                flow_centric_demand_data = create_demand_data(network_load_config=network_load_config,
                                                              eps=eps,
                                                              node_dist=benchmark_dists[benchmark]['node_dist'],
                                                              flow_size_dist=benchmark_dists[benchmark]['flow_size_dist'],
                                                              interarrival_time_dist=benchmark_dists[benchmark]['interarrival_time_dist'],
                                                              num_demands_factor=config.NUM_DEMANDS_FACTOR,
                                                              min_last_demand_arrival_time=config.MIN_LAST_DEMAND_ARRIVAL_TIME,
                                                              auto_node_dist_correction=config.AUTO_NODE_DIST_CORRECTION,
                                                              print_data=False)
                benchmark_demands[benchmark][load][repeat] = flow_centric_demand_data
            end_load = time.time()
            print('Generated \'{}\' demands for load {} of {} in {} seconds.'.format(benchmark, load_counter, num_loads, end_load-start_load))
            load_counter += 1

        end_benchmark = time.time()
        print('Generated demands for benchmark \'{}\' in {} seconds.'.format(benchmark, end_benchmark-start_benchmark))

    end_loops = time.time()
    print('Generated all benchmarks in {} seconds.'.format(end_loops-start_loops))

    print('Saving benchmark data...')
    if path_to_save is not None:
        # save benchmarks
        if save_format == 'json':
            save_data_as_json(path_to_save=path_to_save, data=benchmark_demands, overwrite=False)
        elif save_format == 'csv':
            save_data_as_csv(path_to_save=path_to_save, data=benchmark_demands, overwrite=False)
        else:
            raise Exception('Unrecognised save format \'{}\''.format(save_format))

    print('Finished.')

    return benchmark_demands











    






