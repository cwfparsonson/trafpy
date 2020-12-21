import trafpy
import trafpy.generator as tpg
from trafpy.manager import Demand, RWA, SRPT, DCN, RandomAgent, ParametricAgent, EnvAnalyser
import config

import json
import tensorflow as tf
# tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)
import time



if __name__ == '__main__':
    with tf.device('/cpu'):
        path_to_benchmark_data = '../data/benchmark_data/ndf50_1s_0.4load_university_benchmark_data.json'
        benchmark_data = json.loads(tpg.load_data_from_json(path_to_benchmark_data))

        benchmarks = list(benchmark_data.keys())
        demand_data_list = []
        for benchmark in benchmarks:
            for load in benchmark_data[benchmark]:
                for repeat in benchmark_data[benchmark][load]:
                    demand_data = benchmark_data[benchmark][load][repeat]
                    demand_data_list.append(demand_data)
        demand_data = demand_data_list[0]
        demand = Demand(demand_data=demand_data)

        network = tpg.gen_fat_tree(k=3, N=2, num_channels=1, server_to_rack_channel_capacity=1, rack_to_edge_channel_capacity=5, edge_to_agg_channel_capacity=5, agg_to_core_channel_capacity=5)
        rwa = RWA(tpg.gen_channel_names(config.NUM_CHANNELS), config.NUM_K_PATHS)
        # scheduler = RandomAgent(network, rwa, config.SLOT_SIZE)
        # env = DCN(network, demand, scheduler, config.NUM_K_PATHS, config.SLOT_SIZE, max_flows=config.MAX_FLOWS, max_time=config.MAX_TIME)
        env = DCN(network, demand, None, config.NUM_K_PATHS, config.SLOT_SIZE, max_flows=config.MAX_FLOWS, max_time=config.MAX_TIME)
        scheduler = ParametricAgent(env.observation_space, env.action_space, 2, env=env, Graph=network, RWA=rwa, slot_size=config.SLOT_SIZE)
        print('Registered scheduler')
        raise Exception()

        for episode in range(config.NUM_EPISODES):
            print('\nEpisode {}/{}'.format(episode+1,config.NUM_EPISODES))
            observation = env.reset(config.LOAD_DEMANDS)
            scheduler.register_env(env)
            while True:
                start = time.time()
                action = scheduler.get_action(observation)
                # print('Time: {} | Flows arrived: {} | Flows completed: {} | Flows dropped: {} | Actions: {}'.format(env.curr_time, len(env.arrived_flow_dicts), len(env.completed_flows), len(env.dropped_flows), scheduler.chosen_actions))
                observation, reward, done, info = env.step(action)
                end = time.time()
                print('Time for whole step: {}'.format(end-start))
                if done:
                    print('Episode finished.')
                    analyser = EnvAnalyser(env)
                    analyser.compute_metrics(print_summary=True)
                    break




