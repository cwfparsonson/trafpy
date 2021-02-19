if __name__ == '__main__':
    import trafpy
    from trafpy.benchmarker.tools import gen_benchmark_demands

    import os


    DATA_NAME = 'artificial_light_chancap10_numchans1_mldatNone_bidirectional'
    path_to_save = os.path.dirname(trafpy.__file__)+'/../data/benchmark_data/{}_benchmark_data'.format(DATA_NAME)
    benchmark_demands = gen_benchmark_demands(path_to_save=path_to_save,
                                              save_format='json',
                                              load_prev_dists=False)
