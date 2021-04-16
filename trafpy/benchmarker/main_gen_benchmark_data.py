if __name__ == '__main__':
    import trafpy
    from trafpy.benchmarker.tools import gen_benchmark_demands

    import os


    DATA_NAME = 'rack_dist_sensitivity_0.8_k_4_L_2_n_16_chancap1250_numchans1_mldat3.2e5_bidirectional'
    # DATA_NAME = 'pulse_university_mldatNone'
    # path_to_save = '/scratch/datasets/trafpy/traces/flowcentric/{}_benchmark_data'.format(DATA_NAME)
    path_to_save = '/rdata/ong/trafpy/traces/flowcentric/{}_benchmark_data'.format(DATA_NAME)
    benchmark_demands = gen_benchmark_demands(path_to_save=path_to_save,
                                              save_format='json', # 'json' 'pickle'
                                              separate_files=True,
                                              load_prev_dists=False,
                                              overwrite=False)
