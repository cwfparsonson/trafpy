from trafpy.generator.src.tools import load_data_from_json, unpickle_data, pickle_data
from trafpy.generator.src.demand import Demand
from trafpy.manager.src.simulators.simulators import DCN
from trafpy.manager.src.simulators.env_analyser import EnvAnalyser

import multiprocessing
# from progress.spinner import Spinner
import time
import json
import pickle
import sys
import os
import tensorflow as tf
import glob
from pympler import classtracker
tf.keras.backend.clear_session()


class TestBed: 
    def __init__(self, path_to_benchmark_data):
        if os.path.isdir(path_to_benchmark_data):
            # data split into separate files in a directory
            self.separate_files = True
            self.benchmarks = glob.glob(path_to_benchmark_data + '/*')
        else:
            # all data stored in single file
            self.separate_files = False
            self.benchmark_data = self.load_benchmark_data(path_to_benchmark_data)
            self.benchmarks = list(self.benchmark_data.keys())


        self.reset()

    def reset(self):
        self.envs = multiprocessing.Manager().list()
        self.config = None


    def load_benchmark_data(self, demand_file_path):
        if demand_file_path[-4:] == 'json':
            return json.loads(load_data_from_json(demand_file_path))
        elif demand_file_path[-6:] == 'pickle':
            return unpickle_data(demand_file_path)
        else:
            raise Exception('Unrecognised file type \'{}\''.format(demand_file_path))


    def run_tests(self, config, path_to_save):
        self.config = config

        if self.separate_files:
            # must separate files under common folder
            if os.path.exists(path_to_save):
                # exists, create new version
                version = 2
                while os.path.exists(path_to_save+'_v{}'.format(version)):
                    version += 1
                path_to_save = path_to_save+'_v{}'.format(version)
            else:
                pass
            os.mkdir(path_to_save)
            print('Created directory {} in which to save separate files.'.format(path_to_save))
        else:
            # no need to separate files, save under one file path_to_save
            pass

        # # calc how many jobs to run
        # num_benchmark_tests = 0
        # for benchmark in self.benchmarks:
            # for load in self.benchmark_data[benchmark]:
                # for repeat in self.benchmark_data[benchmark][load]:
                    # num_benchmark_tests += 1
        # num_tests = int(len(config['schedulers']))
        # num_jobs = num_benchmark_tests * num_tests

        jobs = []
        start_time = time.time()
        # _loads = [0.1, 0.2, 0.3, 0.4, 0.5] # DEBUG
        if not self.separate_files:
            for benchmark in self.benchmarks:
                for load in list(self.benchmark_data[benchmark].keys()):
                    for repeat in self.benchmark_data[benchmark][load]:
                        for scheduler in config['schedulers']:
                            if config['loads'] == 'all' or load in config['loads']:
                                # if (json.loads(load) == 0.2 and scheduler.scheduler_name == 'srpt_v2') or (json.loads(load) == 0.2 and scheduler.scheduler_name == 'first_fit'): # DEBUG 
                                # if json.loads(load) == 0.5 and scheduler.scheduler_name == 'srpt_v2': # DEBUG 
                                # if json.loads(load) in _loads: # DEBUG
                                # if json.loads(load) == 0.4: # DEBUG
                                # if scheduler.scheduler_name != 'random' and scheduler.scheduler_name != 'first_fit' and json.loads(load) == 0.4:
                                demand_data = self.benchmark_data[benchmark][load][repeat]
                                demand = Demand(demand_data, config['networks'][0].graph['endpoints'])
                                
                                env = DCN(config['networks'][0], 
                                          demand, 
                                          scheduler,
                                          num_k_paths=config['num_k_paths'],
                                          slot_size=config['slot_size'],
                                          sim_name='benchmark_{}_load_{}_repeat_{}_scheduler_{}'.format(benchmark, load, repeat, scheduler.scheduler_name),
                                          max_flows=config['max_flows'], 
                                          max_time=config['max_time'])

                                p = multiprocessing.Process(target=self.run_test,
                                                            args=(scheduler, env, self.envs, path_to_save,))
                                jobs.append(p)
                                p.start()
        else:
            for benchmark_path in self.benchmarks:
                for scheduler in config['schedulers']:
                    load = float(self.conv_str_path_to_kwarg_value(benchmark_path, 'load_'))
                    benchmark = self.conv_str_path_to_kwarg_value(benchmark_path, 'benchmark_')
                    repeat = self.conv_str_path_to_kwarg_value(benchmark_path, 'repeat_')
                    if config['loads'] == 'all' or load in config['loads']:
                        if scheduler.scheduler_name == 'SRPT': # DEBUG
                            demand = Demand(benchmark_path, config['networks'][0].graph['endpoints'])
                            
                            env = DCN(config['networks'][0], 
                                      demand, 
                                      scheduler,
                                      num_k_paths=config['num_k_paths'],
                                      slot_size=config['slot_size'],
                                      sim_name='benchmark_{}_load_{}_repeat_{}_scheduler_{}'.format(benchmark, load, repeat, scheduler.scheduler_name),
                                      max_flows=config['max_flows'], 
                                      max_time=config['max_time'])

                            p = multiprocessing.Process(target=self.run_test,
                                                        args=(scheduler, env, self.envs, path_to_save+'/'+env.sim_name,))
                            jobs.append(p)
                            p.start()

        for job in jobs:
            job.join() # only execute below code when all jobs finished
        end_time = time.time()
        total_time = round(end_time-start_time, 2)
        # print('\n{} tests completed in {} seconds.'.format(num_jobs, total_time))
        print('Completed all tests completed in {} seconds.'.format(total_time))

        # DEBUG: Don't save if just debugging
        if not self.separate_files:
            # saving all sims as 1 file
            self.save(path=path_to_save, overwrite=False) # save final testbed state
        else:
            # have already saved files separately
            pass
    
    def conv_str_path_to_kwarg_value(self, path, kwarg):
        '''Takes path string containing kwarg_[kwarg_value]_ and returns kwarg_value as a str.'''
        starting_index = len(path) - path[::-1].index('/')
        if kwarg[-1] != '_':
            # should end in under score
            kwarg += '_'
        # print('path: {} | kwarg: {}'.format(path, kwarg))
        for idx in range(starting_index, len(path)):
            # print('idx {}'.format(idx))
            # print(path[idx:idx+len(kwarg)])
            if path[idx:idx+len(kwarg)] == kwarg:
                i = idx+len(kwarg)
                l = ''
                c = path[idx+len(kwarg)]
                while c != '_' and path[idx+len(kwarg)+1:] != '.json':
                    l += c
                    # print('l: {}'.format(l))
                    i += 1
                    c = path[i]
                return l
            else:
                pass

    def run_test(self, scheduler, env, envs, path_to_save):

        tracker = classtracker.ClassTracker()
        tracker.track_object(env)
        tracker.create_snapshot()


        printed_percents = [0]
        observation = env.reset()
        try:
            scheduler.register_env(env)
        except:
            # no need to register env
            pass
        print('starting')
        while True:
            action = scheduler.get_action(observation)
            observation, reward, done, info = env.step(action, print_memory_usage=False, print_processing_time=False)

            # print progress
            flows_arrived, flows_processed = len(env.arrived_flow_dicts), len(env.completed_flows)+len(env.dropped_flows)
            percent_demands_processed = round(100*(flows_processed/env.demand.num_demands), 0)
            if percent_demands_processed % 0.01 == 0: # % 10
                if percent_demands_processed not in printed_percents:
                    percent_demands_arrived = round(100*(flows_arrived/env.demand.num_demands))
                    print('Sim: {} | Flows arrived: {}% | Flows processed: {}%'.format(env.sim_name, percent_demands_arrived, percent_demands_processed))
                    printed_percents.append(percent_demands_processed)
                    
                    tracker.create_snapshot()


            if done:
                # env.get_scheduling_session_summary(print_summary=True)
                print('Completed simulation \'{}\''.format(env.sim_name))
                tracker.stats.print_summary()
                analyser = EnvAnalyser(env)
                analyser.compute_metrics(print_summary=True, 
                                         measurement_start_time='auto',
                                         measurement_end_time='auto')
                try:
                    if self.separate_files:
                        # saving each run separately
                        pickle_data(path_to_save=path_to_save, 
                                    data=env,
                                    overwrite=False, 
                                    zip_data=True, 
                                    print_times=False)
                    else:
                        # saving all in one file
                        envs.append(env) # store env
                except EOFError:
                    print('Memory error appending env to list. See https://stackoverflow.com/questions/57370803/multiprocessing-pool-manager-namespace-eof-error for example. Allocate more system memory or reduce size of TestBed experiment.')
                    sys.exit()
                # self.save(path_to_save, overwrite=True, conv_back_to_mp_manager_list=True) # save curr TestBed state
                break

    def save(self, path, overwrite=False, conv_back_to_mp_manager_list=False):
        start = time.time()
        self.envs = list(self.envs) # conv to list so is picklable
        filename = path + '.obj'

        if overwrite:
            # overwrite prev saved file
            pass
        else:
            # avoid overwriting
            v = 2
            
            while os.path.exists(str(filename)):
                filename = path + '_v{}'.format(v) + '.obj'
                v += 1
        filehandler = open(filename, 'wb')
        pickle.dump(dict(self.__dict__), filehandler)
        filehandler.close()
        end = time.time()
        print('Saved test bench data to {} in {} s'.format(filename, end-start))





















if __name__ == '__main__':
    import os
    import trafpy
    from trafpy.generator.src.networks import gen_fat_tree, gen_channel_names
    from trafpy.manager.src.routers.routers import RWA
    from trafpy.manager.src.schedulers.schedulers import SRPT, SRPT_v2, BASRPT, BASRPT_v2, RandomAgent, FirstFit, FairShare, LambdaShare



    with tf.device('/cpu'):

        # _________________________________________________________________________
        # BASIC CONFIGURATION
        # _________________________________________________________________________
        # DATA_NAME = 'social_media_cloud_k_4_L_2_n_4_chancap500_numchans1_mldat2e6_bidirectional'
        DATA_NAME = 'university_k_4_L_2_n_4_chancap3125_numchans1_mldat2e6_bidirectional'

        # benchmark data
        # path_to_benchmark_data = '/scratch/datasets/trafpy/traces/flowcentric/{}_benchmark_data.json'.format(DATA_NAME)
        path_to_benchmark_data = '/scratch/datasets/trafpy/traces/flowcentric/{}_benchmark_data'.format(DATA_NAME)
        LOADS = [0.1] # 'all' [0.1, 0.2]
        tb = TestBed(path_to_benchmark_data)

        # dcn
        MAX_TIME = 1e5 # None
        # MAX_TIME = 'last_demand_arrival_time'
        MAX_FLOWS = 50 # 10 50 100 500

        # networks
        NUM_CHANNELS = 1
        # networks = [gen_fat_tree(k=4, L=2, n=4, num_channels=NUM_CHANNELS, server_to_rack_channel_capacity=500, rack_to_edge_channel_capacity=1000, edge_to_agg_channel_capacity=1000, agg_to_core_channel_capacity=2000, bidirectional_links=True)]
        networks = [gen_fat_tree(k=4, 
                                 L=2, 
                                 n=4, 
                                 num_channels=NUM_CHANNELS, 
                                 server_to_rack_channel_capacity=500, 
                                 rack_to_core_channel_capacity=1000, 
                                 bidirectional_links=True)]

        # rwas
        NUM_K_PATHS = 2
        # NUM_K_PATHS = 1
        rwas = [RWA(gen_channel_names(NUM_CHANNELS), NUM_K_PATHS)]

        # schedulers
        # SLOT_SIZE = 1e6
        SLOT_SIZE = 50.0 #1e4 1e5 1e2 0.1  1e3 50.0
        PACKET_SIZE = 1 # 300 0.01 1e1 1e2
        schedulers = [SRPT_v2(networks[0], rwas[0], slot_size=SLOT_SIZE, packet_size=PACKET_SIZE),
                      FairShare(networks[0], rwas[0], slot_size=SLOT_SIZE, packet_size=PACKET_SIZE),
                      FirstFit(networks[0], rwas[0], slot_size=SLOT_SIZE, packet_size=PACKET_SIZE),
                      RandomAgent(networks[0], rwas[0], slot_size=SLOT_SIZE, packet_size=PACKET_SIZE)]
        # schedulers = [SRPT(networks[0], rwas[0], slot_size=SLOT_SIZE, packet_size=PACKET_SIZE),
                      # SRPT_v2(networks[0], rwas[0], slot_size=SLOT_SIZE, packet_size=PACKET_SIZE),
                      # BASRPT(networks[0], rwas[0], slot_size=SLOT_SIZE, V=0.1, packet_size=PACKET_SIZE),
                      # FairShare(networks[0], rwas[0], slot_size=SLOT_SIZE, packet_size=PACKET_SIZE)]
        # schedulers = [SRPT(networks[0], rwas[0], slot_size=SLOT_SIZE, packet_size=PACKET_SIZE),
                      # SRPT_v2(networks[0], rwas[0], slot_size=SLOT_SIZE, packet_size=PACKET_SIZE),
                      # BASRPT(networks[0], rwas[0], slot_size=SLOT_SIZE, V=0.1, packet_size=PACKET_SIZE),
                      # BASRPT_v2(networks[0], rwas[0], slot_size=SLOT_SIZE, V=0.1, packet_size=PACKET_SIZE),
                      # FairShare(networks[0], rwas[0], slot_size=SLOT_SIZE, packet_size=PACKET_SIZE)]
        # schedulers = [SRPT_v2(networks[0], rwas[0], slot_size=SLOT_SIZE, packet_size=PACKET_SIZE),
                      # BASRPT_v2(networks[0], rwas[0], slot_size=SLOT_SIZE, V=0.1, packet_size=PACKET_SIZE),
                      # FairShare(networks[0], rwas[0], slot_size=SLOT_SIZE, packet_size=PACKET_SIZE)]
        # schedulers = [BASRPT_v2(networks[0], rwas[0], slot_size=SLOT_SIZE, V=800, packet_size=PACKET_SIZE, scheduler_name='basrpt{}'.format(800)),
                      # BASRPT_v2(networks[0], rwas[0], slot_size=SLOT_SIZE, V=1600, packet_size=PACKET_SIZE, scheduler_name='basrpt{}'.format(1600)),
                      # BASRPT_v2(networks[0], rwas[0], slot_size=SLOT_SIZE, V=3200, packet_size=PACKET_SIZE, scheduler_name='basrpt{}'.format(3200)),
                      # BASRPT_v2(networks[0], rwas[0], slot_size=SLOT_SIZE, V=10000, packet_size=PACKET_SIZE, scheduler_name='basrpt{}'.format(10000))]
        # schedulers = [FairShare(networks[0], rwas[0], slot_size=SLOT_SIZE, packet_size=PACKET_SIZE),
                      # SRPT_v2(networks[0], rwas[0], slot_size=SLOT_SIZE, packet_size=PACKET_SIZE)]
        # _lambdas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        # _lambdas = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
        # _lambdas = [0.1, 0.3, 0.5, 0.7, 0.9]
        # _lambdas = [0.1]
        # for _lambda in _lambdas:
            # schedulers.append(LambdaShare(networks[0], rwas[0], slot_size=SLOT_SIZE, _lambda=_lambda, packet_size=PACKET_SIZE, scheduler_name='\u03BB{}S'.format(_lambda)))
        # schedulers = []
        # Vs = [0.1, 1, 5, 10, 20, 30, 50, 100, 200]
        # for V in Vs:
            # schedulers.append(BASRPT(networks[0], rwas[0], slot_size=SLOT_SIZE, V=V, scheduler_name='V{}_basrpt'.format(V)))



        test_config = {'loads': LOADS,
                       'num_k_paths': NUM_K_PATHS,
                       'max_time': MAX_TIME,
                       'max_flows': MAX_FLOWS,
                       'slot_size': SLOT_SIZE,
                       'packet_size': PACKET_SIZE,
                       'networks': networks,
                       'rwas': rwas,
                       'schedulers': schedulers}

        tb.reset()
        # tb.run_tests(test_config, path_to_save = os.path.dirname(trafpy.__file__)+'/scratch/datasets/trafpy/management/flowcentric/')
        tb.run_tests(test_config, path_to_save = '/scratch/datasets/trafpy/management/flowcentric/{}_testbed_data'.format(DATA_NAME))

        















