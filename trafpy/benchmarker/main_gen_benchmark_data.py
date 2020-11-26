if __name__ == '__main__':
    import trafpy
    from trafpy.benchmarker.tools import gen_benchmark_demands

    import os


    DATA_NAME = 'university_chancap1_mldat6e7'
    path_to_save = os.path.dirname(trafpy.__file__)+'/../data/benchmark_data/{}_benchmark_data'.format(DATA_NAME)
    benchmark_demands = gen_benchmark_demands(path_to_save=path_to_save,
                                              load_prev_dists=False)
