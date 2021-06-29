Creating Custom TrafPy Benchmarks
=================================


It is easy to create your own custom benchmarks in ``TrafPy`` using the ``trafpy.benchmarker.Benchmark``
abstract base class. You need only share your child class code for others to be
able to reproduce the traffic you generated for their own networks.

Import the ``trafpy.benchmarker.Benchmark`` abstract parent class

.. nbplot::

    >>> from trafpy.benchmarker import Benchmark

All flow-centric demands have 3 key properties; flow size, flow interarrival time,
and node distribtuion. Therefore, all child classes inheriting from 
``trafpy.benchmarker.Benchmark`` must define the abstract methods ``get_node_dist``,
``get_interarrival_time_dist``, and ``get_flow_size_dist``.

Inside each of these methods, you should first call the ``trafpy.benchmarker.Benchmark``
parent abstract method to initialise (1) the ``TrafPy`` distribtuion hashtable, 
and (2) the path in which to store the distribution hashtable (automatically decided
by ``TrafPy``). You should then implement your own code to generate the distribution hashtable
(using what you've learned from the previous ``TrafPy`` tutorials and examples on
the GitHub), before finally calling the ``trafpy.benchmarker.Benchmark.save_dist``
abstract method to save the distribution.

For example, consider that you want to create a simple benchmark called ``'my_benchmark'`` with a uniform
node distribution, a constant flow size of 1 information unit, and a constant interarrival 
time of 10 time units::

    import trafpy.generator as tpg

    class MyBenchmark(Benchmark):
        def __init__(self, benchmark_name='my_benchmark', benchmark_version='v001', load_prev_dists=True):
            super(MyBenchmark, self).__init__(benchmark_name, benchmark_version, load_prev_dists)

        def get_node_dist(self, eps, racks_dict=None, dist_name='node_dist'):
            dist, path = super().get_node_dist(eps, racks_dict, dist_name)
            if dist is None or not self.load_prev_dists:
                dist = tpg.gen_uniform_node_dist(eps, show_fig=False, print_data=False)
                super().save_dist(dist, dist_name)
            return dist

        def get_interarrival_time_dist(self, dist_name='interarrival_time_dist'):
            dist, path = super().get_interarrival_time_dist(dist_name)
            if dist is None or not self.load_prev_dists:
                dist = {10: 1}
                super().save_dist(dist, dist_name)
            return dist

        def get_flow_size_dist(self, dist_name='flow_size_dist'):
            dist, path = super().get_flow_size_dist(dist_name)
            if dist is None or not self.load_prev_dists:
                dist = {1: 1}
                super().save_dist(dist, dist_name)
            return dist

You can now use your ``MyBenchmark`` class to generate your ``'my_benchmark'`` 
traffic for any network::

    benchmark = MyBenchmark(benchmark_name='my_benchmark', benchmark_version='v001', load_prev_dists=False)

    # init network
    net = tpg.gen_fat_tree(k=4)

    # generate my_benchmark distributions for this network
    dists = {}
    dists['node_dist'] = benchmark.get_node_dist(net.graph['endpoints'], racks_dict=None, dist_name='node_dist')
    dists['interarrival_time_dist'] = benchmark.get_interarrival_time_dist(dist_name='interarrival_time_dist')
    dists['flow_size_dist'] = benchmark.get_flow_size_dist(dist_name='flow_size_dist')

You can now use the ``trafpy.generator.create_demand_data()`` function as usual
to use these benchmark distributions to generate traffic for your own network.

.. warning::

    Note that to generate distribution data from your custom benchmark, you have
    not used the ``trafpy.benchmarker.BenchmarkImporter`` class. This importer
    class is only for default ``TrafPy`` benchmarks.
