if __name__ == '__main__':
    import trafpy
    from trafpy.benchmarker.tools import gen_benchmark_demands

    import os


    # DATA_NAME = 'university_k_4_L_2_n_4_chancap500_numchans1_mldat2e6_bidirectional'
    DATA_NAME = 'pulse_university_mldatNone'
    # path_to_save = os.path.dirname(trafpy.__file__)+'/scratch/datasets/trafpy/traces/flowcentric/{}_benchmark_data'.format(DATA_NAME)
    path_to_save = '/scratch/datasets/trafpy/traces/flowcentric/{}_benchmark_data'.format(DATA_NAME)
    benchmark_demands = gen_benchmark_demands(path_to_save=path_to_save,
                                              save_format='json', # 'json' 'pickle'
                                              load_prev_dists=False)
