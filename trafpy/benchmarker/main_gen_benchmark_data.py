if __name__ == '__main__':
    import trafpy
    from trafpy.benchmarker.tools import gen_benchmark_demands
    from trafpy.benchmarker import config

    import os

    path_to_save = os.path.dirname(trafpy.__file__)+'/../data/benchmark_data/university_benchmark_data'
    benchmark_demands = gen_benchmark_demands(config,
                                              path_to_save=path_to_save,
                                              load_prev_dists=True)
