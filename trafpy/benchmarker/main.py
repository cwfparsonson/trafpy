if __name__ == '__main__':
    import trafpy
    from trafpy.benchmarker.tools import gen_benchmark_demands
    from trafpy.benchmarker import config

    import os

    benchmark_demands = gen_benchmark_demands(network_capacity=config.NETWORK_CAPACITY,
                                              path_to_save=os.path.dirname(trafpy.__file__)+'/../data',
                                              racks_dict=config.RACKS_DICT,
                                              benchmark_sets=['university'],
                                              num_repeats=1)
