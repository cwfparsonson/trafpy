import trafpy.benchmarker.versions.benchmark_v001 as v001



class BenchmarkImporter:
    def __init__(self, benchmark_version, load_prev_dists=True):
        '''Imports pre-defined TrafPy benchmark distributions for a network.

        Args:
            benchmark_version (str): TrafPy benchmark version to access (e.g. '0.0.1').
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

        self.valid_versions = ['0.0.1']

        if benchmark_version not in self.valid_versions:
            raise Exception('Unrecognised benchmark version \'{}\'. Please enter a valid benchmark version (one of:\n{}).'.format(benchmark_version, self.valid_versions))

        self.benchmark_version = benchmark_version
        
        if self.benchmark_version == '0.0.1':
            from trafpy.benchmarker.versions.benchmark_v001.distribution_generator import DistributionGenerator
            self.distgen = DistributionGenerator(load_prev_dists=self.load_prev_dists)
        
        self.valid_benchmark_sets = self.distgen.valid_benchmark_sets



    def get_benchmark_dists(self, benchmark, eps, racks_dict=None):
        '''Retrieves pre-defined TrafPy benchmark distributions for a network.

        Args:
            benchmark (str): Name of benchmark (e.g. 'university').
            eps (list): List of end points/machines/leaf nodes in network.
            racks_dict (dict): Mapping of which end points are in which racks. Keys are
                rack ids, values are list of end points. If None, assume there is not
                clustering/rack system in the network where have different end points
                in different clusters/racks.
        '''
        if benchmark not in self.valid_benchmark_sets:
            raise Exception('Unrecognised benchmark set \'{}\'. Valid benchmark sets for benchmark {}:\n{}'.format(benchmark, self.benchmark_version, self.valid_benchmark_sets))

        benchmark_dists = {'node_dist': self.distgen.get_node_dist(benchmark, racks_dict, eps),
                           'flow_size_dist': self.distgen.get_flow_size_dist(benchmark),
                           'interarrival_time_dist': self.distgen.get_interarrival_time_dist(benchmark),
                           'num_ops_dist': self.distgen.get_num_ops_dist(benchmark)}

        return benchmark_dists



