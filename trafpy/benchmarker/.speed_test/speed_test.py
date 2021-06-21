'''
Measures speed of read/write from/to databases in specified locations.
Useful for benchmarking different memory systems for holding databases.
N.B. Must have speed_test_database.sqlite in same dir as this speed_test.py file.

'''
import time
import os
import shutil
from sqlitedict import SqliteDict
import numpy as np
from IPython.display import display
import pandas as pd
from tqdm import tqdm



def init_database_in_test_dir(test_directory, database_name):
    '''
    Copies database_name database from current directory to test_directory so
    that can perform speed tests on database_name database in test_directory.

    '''
    cwd = os.getcwd()
    database_path = cwd + '/' + database_name
    new_database_name = 'new_' + database_name
    new_database_path = test_directory + '/' + new_database_name
    shutil.copyfile(database_path, new_database_path)
    return new_database_name

def perform_copy_paste_test(test_directory, num_repeats, database_name):
    cwd = os.getcwd()
    database_path = cwd + '/' + database_name
    new_database_path = test_directory + '/copy_' + database_name

    # init progress bar
    pbar = tqdm(total=num_repeats,
                desc='Copy test',
                leave=False,
                smoothing=0)

    results = []
    for _ in range(num_repeats):
        start = time.time()
        shutil.copyfile(database_path, new_database_path)
        end = time.time()
        results.append(end-start)
        pbar.update(1)
    pbar.close()
    return np.mean(results)

def perform_read_test(test_directory, num_repeats, database_name):
    database_path = test_directory + '/' + database_name

    # init progress bar
    with SqliteDict(database_path) as database:
        num_keys = len(list(database.keys()))
        database.close()
    pbar = tqdm(total=num_repeats*num_keys,
                desc='Read test',
                leave=False,
                smoothing=0)

    # run tests
    results = []
    for _ in range(num_repeats):
        start = time.time()
        storage = {}
        with SqliteDict(database_path) as database:
            # slot_size = database['slot_size']
            # job_centric = database['job_centric']
            # num_demands = database['num_demands']
            # num_flows = database['num_flows']
            for key, val in database.items():
                storage[key] = val
                pbar.update(1)
            database.close()
        end = time.time()
        results.append(end-start)
    pbar.close()
    return np.mean(results)

def perform_write_test(test_directory, num_repeats, database_name):
    database_path = test_directory + '/' + database_name

    # init progress bar
    with SqliteDict(database_path) as database:
        num_keys = len(list(database.keys()))
        database.close()
    pbar = tqdm(total=num_repeats*num_keys,
                desc='Write test',
                leave=False,
                smoothing=0)

    results = []
    for _ in range(num_repeats):
        start = time.time()
        with SqliteDict(database_path) as database:
            # database['slot_size'] = 1e4
            # database['job_centric'] = True
            # database['num_demands'] = 10e3
            # database['num_flows'] = 10e4
            for key in database.keys():
                database[key] = None
                pbar.update(1)
            database.commit()
            database.close()
        end = time.time()
        results.append(end-start)
    pbar.close()
    return np.mean(results)



def perform_speed_tests(test_directory, num_repeats, database_name='speed_test_database.sqlite'):
    '''
    Measures speed of read/write/copy to/from test_directory num_repeats times
    and returns average results.
    '''
    results = {}

    print('\nPerforming speed tests in directory {} for database {}'.format(test_directory, database_name))

    # establish database in test_directory
    new_database_name = init_database_in_test_dir(test_directory, database_name)
    print('Initialised {} in {}\nBeginning speed tests...'.format('new_'+database_name, test_directory))

    # copy-paste database test
    copy_paste_result = perform_copy_paste_test(test_directory, num_repeats, new_database_name)
    print('Copy-paste database speed result: {} s.'.format(copy_paste_result))

    # read database test
    read_result = perform_read_test(test_directory, num_repeats, new_database_name)
    print('Read database speed result: {} s.'.format(read_result))

    # write database test
    write_result = perform_write_test(test_directory, num_repeats, new_database_name)
    print('Write database speed result: {} s.'.format(write_result))

    results['copy'] = copy_paste_result
    results['read'] = read_result
    results['write'] = write_result
    return results







if __name__ == '__main__':
    # enter params
    num_repeats = 5
    # EEE servers:
    # test_directories = {'/rdata': '/rdata/ong/trafpy/.speed_test/',
                        # '/scratch': '/scratch/datasets/trafpy/.speed_test/',
                        # '/local': '/home/zciccwf/local/.speed_test/',
                        # 'cwd': '/home/zciccwf/phd_project/projects/trafpy/trafpy/benchmarker/.speed_test/',
                        # '/space': '/space/ONG/trafpy/.speed_test/'}
    test_directories = {'/rdata': '/rdata/ong/trafpy/.speed_test/',
                        '/scratch': '/scratch/datasets/trafpy/.speed_test/',
                        '/local': '/home/zciccwf/local/.speed_test/',
                        'cwd': '/home/zciccwf/phd_project/projects/trafpy/trafpy/benchmarker/.speed_test/'}
    # UCL HPC servers:

    # run tests
    start = time.time()
    summary_table = {'test': [], 'copy': [], 'read': [], 'write': []}
    for test_name, test_directory in test_directories.items():
        results = perform_speed_tests(test_directory, num_repeats, database_name='speed_test_database.sqlite')

        # update summary table
        summary_table['test'].append(test_name)
        for key, val in results.items():
            summary_table[key].append(val)
    end = time.time()
    print('\nCompleted tests in {} s'.format(end-start))

    # display summary table in terminal
    summary_table_df = pd.DataFrame(summary_table, index=None)
    print('\nSummary table:\n')
    display(summary_table_df)

