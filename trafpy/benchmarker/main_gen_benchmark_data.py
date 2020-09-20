if __name__ == '__main__':
    import trafpy
    from trafpy.benchmarker.tools import gen_benchmark_demands

    import os


    path_to_save = os.path.dirname(trafpy.__file__)+'/../data/benchmark_data/social_media_cloud_benchmark_data'
    benchmark_demands = gen_benchmark_demands(path_to_save=path_to_save,
                                              load_prev_dists=False)
