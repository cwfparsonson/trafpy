from trafpy.generator.src.builder import create_demand_data
from trafpy.generator.src.tools import save_data_as_json
from trafpy.benchmarker import config
from trafpy.benchmarker.versions.benchmark_importer import BenchmarkImporter

import numpy as np
import time






def gen_benchmark_demands(network_capacity, 
                          path_to_save=None,
                          load_prev_dists=True,
                          racks_dict=None,
                          loads=np.arange(0.1, 1.1, 0.1).tolist(),
                          benchmark_version='0.0.1', 
                          benchmark_sets=['all'], 
                          num_repeats=10):
    # remove python floating point arithmetic errors in loads
    loads = [round(load, 2) for load in loads]

    # init benchmark importer
    importer = BenchmarkImporter(benchmark_version, load_prev_dists=load_prev_dists)

    if racks_dict is None:
        print('No racks_dict given. Loading racks_dict from config.py...')
        racks_dict = config.RACKS_DICT
        print('Loaded racks dict:\n{}'.format(racks_dict))

    # get endpoint labels
    eps_racks_list = [eps for eps in racks_dict.values()]
    eps = []
    for rack in eps_racks_list:
        for ep in rack:
            eps.append(ep)

    if benchmark_sets == ['all']:
        benchmark_sets = importer.valid_benchmark_sets

    benchmark_dists = {benchmark: {} for benchmark in benchmark_sets}
    benchmark_demands = {benchmark: {load: {repeat: {} for repeat in range(num_repeats)} for load in loads} for benchmark in benchmark_sets}
    num_loads = len(loads)
    start_loops = time.time()
    print('\n~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*')
    print('Benchmarks to Generate: {}'.format(benchmark_sets))
    print('Loads to generate: {}'.format(loads))
    print('Number of repeats to generate for each benchmark load: {}'.format(num_repeats))
    for benchmark in benchmark_sets:
        print('~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*')
        print('Generating demands for benchmark \'{}\'...'.format(benchmark))
        start_benchmark = time.time()
        load_counter = 1
        benchmark_dists[benchmark] = importer.get_benchmark_dists(benchmark, racks_dict, eps)
        for load in loads:
            start_load = time.time()
            network_load_config = {'network_rate_capacity': network_capacity, 
                                   'target_load_fraction': load}
            for repeat in range(num_repeats):
                flow_centric_demand_data = create_demand_data(network_load_config=network_load_config,
                                                              eps=eps,
                                                              node_dist=benchmark_dists[benchmark]['node_dist'],
                                                              flow_size_dist=benchmark_dists[benchmark]['flow_size_dist'],
                                                              interarrival_time_dist=benchmark_dists[benchmark]['interarrival_time_dist'],
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
        save_data_as_json(path_to_save=path_to_save, data=benchmark_demands, overwrite=False)

    print('Finished.')

    return benchmark_demands











    






