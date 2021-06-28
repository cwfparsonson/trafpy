import importlib



class BenchmarkImporter:
    def __init__(self, benchmark_version='v001', load_prev_dists=True):
        '''Imports pre-defined TrafPy benchmark distributions for a network.

        Args:
            benchmark_version (str): TrafPy benchmark version to access (e.g. 'v001').
            load_prev_dists (bool): If True, will generate a new benchmark distribution
                for the network(s) you provide the imported. This is needed if
                you have a network with a different number of endpoints or with
                different end point labels. If False, will load the exact same distributions as was previously
                defined, which is needed if you want to use the exact same benchmark
                distribution multiple times.

        '''
        self.load_prev_dists = load_prev_dists
        if self.load_prev_dists:
            print('load_prev_dists=True. Where possible, will load previously generated distributions. Ensure that e.g. network endpoints, rack prob configs etc. are correct. To overwrite saved distributions with your current configuration, set load_prev_dists to False.')
        else:
            print('load_prev_dist=False. Will re-generate dists with given network params and override any previously saved distributions.')

        self.valid_versions = ['v001']

        if benchmark_version not in self.valid_versions:
            raise Exception('Unrecognised benchmark version \'{}\'. Please enter a valid benchmark version (one of:\n{}).'.format(benchmark_version, self.valid_versions))

        self.benchmark_version = benchmark_version
        
        if self.benchmark_version == 'v001':
            from trafpy.benchmarker.versions.benchmark_v001.config import get_default_benchmark_names
        self.default_benchmark_names = get_default_benchmark_names()

    def get_benchmark_dists(self, benchmark_name, eps, racks_dict=None):
        '''Retrieves pre-defined TrafPy benchmark distributions for a network.

        Args:
            benchmark_name (str): Name of benchmark (e.g. 'university').
            eps (list): List of end points/machines/leaf nodes in network.
            racks_dict (dict): Mapping of which end points are in which racks. Keys are
                rack ids, values are list of end points. If None, assume there is not
                clustering/rack system in the network where have different end points
                in different clusters/racks.
        '''
        if benchmark_name not in self.default_benchmark_names:
            raise Exception('Unrecognised default benchmark set \'{}\'. Valid default benchmark sets for benchmark version {}:\n{}'.format(benchmark_name, self.benchmark_version, self.default_benchmark_names))

        # import benchmark class and instantiate benchmark object
        benchmark_module = importlib.import_module('trafpy.benchmarker.versions.benchmark_{}.benchmarks.{}'.format(self.benchmark_version, benchmark_name))
        benchmark = benchmark_module.DefaultBenchmark(benchmark_name=benchmark_name, benchmark_version=self.benchmark_version, load_prev_dists=self.load_prev_dists)

        # get benchmark dists
        benchmark_dists = {'node_dist': benchmark.get_node_dist(eps, racks_dict, dist_name='node_dist'),
                           'flow_size_dist': benchmark.get_flow_size_dist(dist_name='flow_size_dist'),
                           'interarrival_time_dist': benchmark.get_interarrival_time_dist(dist_name='interarrival_time_dist')}
        if benchmark.jobcentric:
            benchmark_dists['num_ops_dist'] = benchmark.get_num_ops_dist(dist_name='num_ops_dist')
        else:
            benchmark_dists['num_ops_dist'] = None

        return benchmark_dists



