from trafpy.generator.src.tools import load_data_from_json
from trafpy.generator.src.demand import Demand
from trafpy.manager.src.simulators.simulators import DCN

import multiprocessing
# from progress.spinner import Spinner
import time
import json
import pickle


class TestBed:

    def __init__(self, path_to_benchmark_data):
        self.benchmark_data = self.load_benchmark_data(path_to_benchmark_data)
        self.benchmarks = list(self.benchmark_data.keys())

        self.reset()

    def reset(self):
        self.envs = multiprocessing.Manager().list()
        self.config = None

    def load_benchmark_data(self, demand_file_path):
        return json.loads(load_data_from_json(demand_file_path))


    def run_tests(self, config, path_to_save):
        self.config = config

        # calc how many jobs to run
        num_benchmark_tests = 0
        for benchmark in self.benchmarks:
            for load in self.benchmark_data[benchmark]:
                for repeat in self.benchmark_data[benchmark][load]:
                    num_benchmark_tests += 1
        num_tests = int(len(config['schedulers']))
        num_jobs = num_benchmark_tests * num_tests

        jobs = []
        start_time = time.time()
        for benchmark in self.benchmarks:
            # for load in self.benchmark_data[benchmark]:
            for load in list(self.benchmark_data[benchmark].keys())[:2]:
                for repeat in self.benchmark_data[benchmark][load]:
                    for scheduler in config['schedulers']:
                        demand_data = self.benchmark_data[benchmark][load][repeat]
                        demand = Demand(demand_data)
                        env = DCN(config['networks'][0], 
                                  demand, 
                                  scheduler, 
                                  slot_size=config['slot_size'],
                                  max_flows=config['max_flows'], 
                                  max_time=config['max_time'])
                        self.run_test(scheduler, env, self.envs)
                        p = multiprocessing.Process(target=self.run_test,
                                                    args=(scheduler, env, self.envs, path_to_save,))
                        jobs.append(p)
                        p.start()
        for job in jobs:
            job.join() # only execute below code when all jobs finished
        end_time = time.time()
        total_time = round(end_time-start_time, 2)
        print('\n{} tests completed in {} seconds.'.format(num_jobs, total_time))

        self.save(path=path_to_save, overwrite=True, conv_back_to_mp_manager_list=False) # save final testbed state


    def run_test(self, scheduler, env, envs, path_to_save):
        observation = env.reset()
        while True:
            action = scheduler.get_action(observation)
            observation, reward, done, info = env.step(action)
            if done:
                env.get_scheduling_session_summary(print_summary=True)
                envs.append(env) # store env
                self.save(path_to_save, overwrite=True, conv_back_to_mp_manager_list=True) # save curr TestBed state
                break

    def save(self, path, conv_back_to_mp_manager_list=False):
        self.envs = list(self.envs) # conv to list so is picklable
        filename = path + self.config['test_name'] + '.obj'
        filehandler = open(filename, 'wb')
        pickle.dump(dict(self.__dict__), filehandler)
        filehandler.close()

        if conv_back_to_mp_manager_list:
            self.envs = multiprocessing.Manager().list(self.envs) # conv back to list to continue multiprocessing if needed




















if __name__ == '__main__':
    import os
    import trafpy
    from trafpy.generator.src.networks import gen_fat_tree, gen_channel_names
    from trafpy.manager.src.routers.routers import RWA
    from trafpy.manager.src.schedulers.schedulers import SRPT, BASRPT




    # _________________________________________________________________________
    # BASIC CONFIGURATION
    # _________________________________________________________________________
    MAX_TIME = None


    path_to_benchmark_data = os.path.dirname(trafpy.__file__)+'/../data/benchmark_data/university_benchmark_data.json'
    tb = TestBed(path_to_benchmark_data)

    # networks
    NUM_CHANNELS = 1
    networks = [gen_fat_tree(k=4, N=30, num_channels=NUM_CHANNELS)]

    # rwas
    NUM_K_PATHS = 2
    rwas = [RWA(gen_channel_names(NUM_CHANNELS), NUM_K_PATHS)]

    # schedulers
    SLOT_SIZE = 1e6 
    schedulers = [SRPT(networks[0], rwas[0], slot_size=SLOT_SIZE),
                  BASRPT(networks[0], rwas[0], slot_size=SLOT_SIZE, V=40)]



    test_config = {'test_name': 'university_benchmark_test_1',
                   'max_time': MAX_TIME,
                   'max_flows': None,
                   'slot_size': SLOT_SIZE,
                   'networks': networks,
                   'rwas': rwas,
                   'schedulers': schedulers}

    tb.reset()
    tb.run_tests(test_config, path_to_save = os.path.dirname(trafpy.__file__+'/../data/'))

    














