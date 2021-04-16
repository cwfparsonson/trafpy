import time
import os
import shutil
from sqlitedict import SqliteDict
import numpy as np


if __name__ == '__main__':
    num_repeats = 10
    # use_scratch = False

    # # processing speed
    # results = []
    # for _ in range(num_repeats):
        # start = time.time()
        # lst = []
        # for i in range(1000000):
            # lst.append('x')
        # end = time.time()
        # results.append(end-start)
    # result = np.mean(results)
    # print('Processing speed result: {} s.'.format(result))

    # # create database
    # cwd = os.getcwd()
    # if use_scratch:
        # database_path = '/scratch/datasets/trafpy/database_test.sqlite'
    # else:
        # database_path = 'database_test.sqlite'
    # results = []
    # for _ in range(num_repeats):
        # start = time.time()
        # with SqliteDict(database_path) as database:
            # for key, val in enumerate(lst[:10000]):
                # database[key] = val
            # database.commit()
            # database.close()
        # end = time.time()
        # results.append(end-start)
    # result = np.mean(results)
    # print('Create database speed result: {} s.'.format(result))

    # # read from database
    # results = []
    # for _ in range(num_repeats):
        # read_dict = {}
        # start = time.time()
        # with SqliteDict(database_path) as database:
            # for key, val in database.items():
                # read_dict[key] = val
        # end = time.time()
        # results.append(end-start)
    # result = np.mean(results)
    # print('Read database speed result: {} s.'.format(result))

    # # copy database
    # new_database_path = database_path[:-7] + '_copy.sqlite'
    # results = []
    # for _ in range(num_repeats):
        # start = time.time()
        # shutil.copyfile(database_path, new_database_path)
        # end = time.time()
        # results.append(end-start)
    # result = np.mean(results)
    # print('Copying database speed result: {} s.'.format(result))










    # cwd
    # copy database
    cwd = os.getcwd()
    database_path = cwd + '/database_test.sqlite'
    # new_database_path = '/scratch/datasets/trafpy/new_database_test.sqlite'
    new_database_path = cwd + '/new_database_test.sqlite'
    results = []
    for _ in range(num_repeats):
        start = time.time()
        shutil.copyfile(database_path, new_database_path)
        end = time.time()
        results.append(end-start)
    result = np.mean(results)
    print('Copy cwd database speed result: {} s.'.format(result))

    # read from database
    results = []
    for _ in range(num_repeats):
        read_dict = {}
        start = time.time()
        with SqliteDict(database_path) as database:
            # for key, val in database.items():
                # read_dict[key] = val
            slot_size = database['slot_size']
            job_centric = database['job_centric']
            num_demands = database['num_demands']
            num_flows = database['num_flows']
        end = time.time()
        results.append(end-start)
    result = np.mean(results)
    print('Read cwd database speed result: {} s.'.format(result))



    # /scratch
    # copy database
    database_path = '/scratch/datasets/trafpy/traces/flowcentric/skewed_nodes_sensitivity_0.2_k_4_L_2_n_16_chancap1250_numchans1_mldat3.2e5_bidirectional_benchmark_data/benchmark_skewed_nodes_sensitivity_0.2_load_0.9_repeat_0_slotsize_1000.0_slots_dict.sqlite'
    new_database_path = '/scratch/datasets/trafpy/new_database_test.sqlite'
    results = []
    for _ in range(num_repeats):
        start = time.time()
        shutil.copyfile(database_path, new_database_path)
        end = time.time()
        results.append(end-start)
    result = np.mean(results)
    print('Copy /scratch database speed result: {} s.'.format(result))

    # read from database
    results = []
    for _ in range(num_repeats):
        read_dict = {}
        start = time.time()
        with SqliteDict(database_path) as database:
            # for key, val in database.items():
                # read_dict[key] = val
            slot_size = database['slot_size']
            job_centric = database['job_centric']
            num_demands = database['num_demands']
            num_flows = database['num_flows']
        end = time.time()
        results.append(end-start)
    result = np.mean(results)
    print('Read /scratch database speed result: {} s.'.format(result))


    # /rdata
    # copy database
    database_path = '/rdata/ong/trafpy/database_test.sqlite'
    new_database_path = '/rdata/ong/trafpy/new_database_test.sqlite'
    results = []
    for _ in range(num_repeats):
        start = time.time()
        shutil.copyfile(database_path, new_database_path)
        with SqliteDict(database_path) as database:
            database.close()
        end = time.time()
        results.append(end-start)
    result = np.mean(results)
    print('Copy /rdata database speed result: {} s.'.format(result))

    # read /rdata from database
    results = []
    for _ in range(num_repeats):
        read_dict = {}
        start = time.time()
        with SqliteDict(database_path) as database:
            # for key, val in database.items():
                # read_dict[key] = val
            slot_size = database['slot_size']
            job_centric = database['job_centric']
            num_demands = database['num_demands']
            num_flows = database['num_flows']
        end = time.time()
        results.append(end-start)
    result = np.mean(results)
    print('Read /rdata database speed result: {} s.'.format(result))




