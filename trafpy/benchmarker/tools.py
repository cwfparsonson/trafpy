from trafpy.benchmarker import config
from trafpy.generator.src.builder import create_demand_data
from trafpy.generator.src.tools import save_data_as_json, save_data_as_csv, pickle_data
from trafpy.benchmarker.versions.benchmark_importer import BenchmarkImporter
from trafpy.generator.src.demand import Demand

import numpy as np
import time
import os
from collections import defaultdict # use for initialising arbitrary length nested dict
from sqlitedict import SqliteDict
import json




def gen_benchmark_demands(path_to_save=None, 
                          save_format='json', 
                          separate_files=False,
                          load_prev_dists=True,
                          overwrite=False):
    '''
    If separate_files, will save each load, repeat and, and benchmark to separate
    files in a common folder. This can help with memory since not storing everything
    in one large file.

    If slot size is not None, will also generate an sqlite database for the slots_dict
    dictionary. This is useful if later during simulations want to have pre-computed
    slots_dict rather than computing & storing them in memory.

    '''
    if path_to_save[-1] == '/' or path_to_save[-1] == '\\':
        path_to_save = path_to_save[:-1]

    if separate_files:
        # must separate files under common folder
        if os.path.exists(path_to_save):
            if not overwrite:
                # exists, do not overwrite, create new version
                version = 2
                while os.path.exists(path_to_save+'_v{}'.format(version)):
                    version += 1
                path_to_save = path_to_save+'_v{}'.format(version)
                os.mkdir(path_to_save)
            else:
                # overwrite
                answer = ''
                while answer not in ['y', 'n']:
                    answer = input('Overwrite data in {}? (y/n)'.format(path_to_save)).lower()
                if answer == 'n':
                    raise Exception('User does not want to overwrite data, terminating session.')
        else:
            # directory doesn't exist, must make
            os.mkdir(path_to_save)
        print('Set directory {} in which to save separate files.'.format(path_to_save))
    else:
        # no need to separate files, save under one file path_to_save
        pass

    # init benchmark importer
    importer = BenchmarkImporter(config.BENCHMARK_VERSION, load_prev_dists=load_prev_dists)

    # load distributions for each benchmark
    benchmark_dists = {benchmark: {} for benchmark in config.BENCHMARKS}

    nested_dict = lambda: defaultdict(nested_dict)
    benchmark_demands = nested_dict()

    # begin generating data for each benchmark
    num_loads = len(config.LOADS)
    start_loops = time.time()
    print('\n~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*')
    print('Benchmarks to Generate: {}'.format(config.BENCHMARKS))
    print('Loads to generate: {}'.format(config.LOADS))
    print('Number of sets to generate for each benchmark load: {}'.format(config.NUM_REPEATS))
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
        benchmark_dists[benchmark] = importer.get_benchmark_dists(benchmark, eps, racks_dict=racks_dict)
        for load in config.LOADS:
            start_load = time.time()
            network_load_config = {'network_rate_capacity': config.NETWORK_CAPACITIES[benchmark], 
                                   'ep_link_capacity': config.NETWORK_EP_LINK_CAPACITIES[benchmark],
                                   'target_load_fraction': load,
                                   'disable_timeouts': True}
            for repeat in range(config.NUM_REPEATS):
                print('Generating demand data for benchmark {} load {} repeat {}...'.format(benchmark, load, repeat))
                if benchmark_dists[benchmark]['num_ops_dist'] is not None:
                    # job-centric
                    use_multiprocessing = True
                else:
                    # flow-centric
                    use_multiprocessing = False
                demand_data = create_demand_data(min_num_demands=config.MIN_NUM_DEMANDS,
                                                 max_num_demands=config.MAX_NUM_DEMANDS, 
                                                 eps=eps,
                                                 node_dist=benchmark_dists[benchmark]['node_dist'],
                                                 flow_size_dist=benchmark_dists[benchmark]['flow_size_dist'],
                                                 interarrival_time_dist=benchmark_dists[benchmark]['interarrival_time_dist'],
                                                 num_ops_dist=benchmark_dists[benchmark]['num_ops_dist'],
                                                 c=3, # 1.5
                                                 jensen_shannon_distance_threshold=config.JENSEN_SHANNON_DISTANCE_THRESHOLD,
                                                 network_load_config=network_load_config,
                                                 min_last_demand_arrival_time=config.MIN_LAST_DEMAND_ARRIVAL_TIME,
                                                 auto_node_dist_correction=config.AUTO_NODE_DIST_CORRECTION,
                                                 use_multiprocessing=use_multiprocessing,
                                                 print_data=False)
                if separate_files:
                    print('Saving demand data for benchmark {} load {} repeat {}...'.format(benchmark, load, repeat))
                    # save as benchmark, load, and repeat into separate files
                    file_path = path_to_save + '/benchmark_{}_load_{}_repeat_{}'.format(benchmark, load, repeat)
                    if save_format == 'json':
                        save_data_as_json(path_to_save=file_path, data=demand_data, overwrite=overwrite)
                    elif save_format == 'csv':
                        save_data_as_csv(path_to_save=file_path, data=demand_data, overwrite=overwrite)
                    elif save_format == 'pickle':
                        pickle_data(path_to_save=file_path, data=demand_data, overwrite=overwrite)
                    else:
                        raise Exception('Unrecognised save format \'{}\''.format(save_format))
                    # reset benchmark demands dict to save memory
                    benchmark_demands = nested_dict()

                else:
                    # saving all benchmarks, loads and repeats into one file
                    benchmark_demands[benchmark][load][repeat] = demand_data

                if config.SLOT_SIZE is not None:
                    # generate slots dict and save as database
                    print('Creating slots_dict database with slot_size {}...'.format(config.SLOT_SIZE))
                    s = time.time()
                    demand = Demand(demand_data, eps=eps)
                    with SqliteDict(file_path+'_slotsize_{}_slots_dict.sqlite'.format(config.SLOT_SIZE)) as slots_dict:
                        for key, val in demand.get_slots_dict(slot_size=config.SLOT_SIZE, include_empty_slots=True, print_info=True).items():
                            if type(key) is not str:
                                slots_dict[json.dumps(key)] = val
                            else:
                                slots_dict[key] = val
                        slots_dict.commit()
                        slots_dict.close()
                    e = time.time()
                    print('Created slots_dict database in {} s'.format(e-s))
                else:
                    pass

            end_load = time.time()
            print('Generated \'{}\' demands for load {} of {} in {} seconds.'.format(benchmark, load_counter, num_loads, end_load-start_load))
            load_counter += 1

        end_benchmark = time.time()
        print('Generated demands for benchmark \'{}\' in {} seconds.'.format(benchmark, end_benchmark-start_benchmark))

    end_loops = time.time()
    print('Generated all benchmarks in {} seconds.'.format(end_loops-start_loops))

    if not separate_files:
        # save all benchmarks, loads, and repeats into 1 file
        print('Saving benchmark data...')
        if path_to_save is not None:
            # save benchmarks
            if save_format == 'json':
                save_data_as_json(path_to_save=path_to_save, data=benchmark_demands, overwrite=overwrite)
            elif save_format == 'csv':
                save_data_as_csv(path_to_save=path_to_save, data=benchmark_demands, overwrite=overwrite)
            elif save_format == 'pickle':
                pickle_data(path_to_save=path_to_save, data=benchmark_demands, overwrite=overwrite)
            else:
                raise Exception('Unrecognised save format \'{}\''.format(save_format))

    print('Finished.')

    return benchmark_demands











    






