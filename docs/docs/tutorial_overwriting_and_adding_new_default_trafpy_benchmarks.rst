Overwriting and Adding New Default TrafPy Benchmarks
====================================================

.. warning::
    
    This tutorial is for advanced ``TrafPy`` developers who want to commit their 
    own benchmarks to the community by adding them to the set of default
    ``TrafPy`` benchmarks. If you simply want to make your own benchmark but not
    make it a default ``TrafPy`` benchmark, see the previous tutorial.

``TrafPy`` benchmarks are stored in the ``benchmarks/`` directory of whereever
``TrafPy`` is installed on your machine. You can check this with:

.. nbplot::

    >>> import os
    >>> import trafpy
    >>> benchmarks_path = os.path.dirname(trafpy.__file__) + '/benchmarker/versions/benchmark_v001/'
    >>> print(benchmarks_path)
    /home/cwfparsonson/Insync/zciccwf@ucl.ac.uk/OneDriveBiz/ipes_cdt/phd_project/projects/trafpy/trafpy/benchmarker/versions/benchmark_v001/

Inside this directory will be a 2 key folders: the ``benchmarks/`` folder, which
is where the ``trafpy.benchmarker.Benchmark`` child classes are stored
for the default ``TrafPy`` benchmarks, and the ``data/`` folder which is where
any generated distribution hashtables are stored for each benchmark.

To add a new benchmark, simply open the ``benchmarks/`` folder and create a python
file called ``<benchmark_name>.py``.

.. warning::

    The name of the file will determine the name of your default benchmark. E.g.
    to create a benchmark called 'university', you must create a file called
    ``university.py``.

Inside this file, you can write your own default benchmark class inheriting
from the ``trafpy.benchmarker.Benchmark`` abstract parent class. The only constraint
is that your class **must** be called ``DefaultBenchmark``; failing to set the class
variable to this will result in errors with the ``trafpy.benchmarker.BenchmarkImporter``
class.

.. warning::

    Your default benchmark class variable **must** be called ``DefaultBenchmark``.

For example, this is how you might create a new default benchmark class called
``'my_default_benchmark'`` in ``benchmarks_path+'/benchmarks/my_default_benchmark.py'``::

    from trafpy.benchmarker.versions.benchmark import Benchmark
    from trafpy.generator.src.dists import node_dists
    from trafpy.generator.src.dists import val_dists
    from trafpy.generator.src.dists import plot_dists

    import math
    import numpy as np

    class DefaultBenchmark(Benchmark):
        def __init__(self, benchmark_name='my_default_benchmark', benchmark_version='v001', load_prev_dists=True):
            super(DefaultBenchmark, self).__init__(benchmark_name, benchmark_version, load_prev_dists)

        def get_node_dist(self, eps, racks_dict=None, dist_name='node_dist'):
            dist, path = super().get_node_dist(eps, racks_dict, dist_name)
            if dist is None or not self.load_prev_dists:
                num_skewed_nodes = math.ceil(0.2 * len(eps))
                skewed_node_probs = [0.55/num_skewed_nodes for _ in range(num_skewed_nodes)]
                if racks_dict is None:
                    rack_prob_config = None
                else:
                    rack_prob_config = {'racks_dict': racks_dict, 'prob_inter_rack': 0.7}
                dist = node_dists.gen_multimodal_node_dist(eps, 
                                                            rack_prob_config=rack_prob_config, 
                                                            num_skewed_nodes=num_skewed_nodes, 
                                                            skewed_node_probs=skewed_node_probs, 
                                                            show_fig=False, 
                                                            print_data=False)
                super().save_dist(dist, dist_name)
            return dist

        def get_interarrival_time_dist(self, dist_name='interarrival_time_dist'):
            dist, path = super().get_interarrival_time_dist(dist_name)
            if dist is None or not self.load_prev_dists:
                dist = val_dists.gen_named_val_dist(dist='weibull',
                                                    params={'_alpha': 0.9, '_lambda': 6000},
                                                    min_val=1,
                                                    round_to_nearest=25,
                                                    show_fig=False,
                                                    print_data=False)
                super().save_dist(dist, dist_name)
            return dist

        def get_flow_size_dist(self, dist_name='flow_size_dist'):
            dist, path = super().get_flow_size_dist(dist_name)
            if dist is None or not self.load_prev_dists:
                dist = val_dists.gen_named_val_dist(dist='lognormal',
                                                    params={'_mu': 7, '_sigma': 2.5},
                                                    min_val=1,
                                                    max_val=2e7,
                                                    round_to_nearest=25,
                                                    show_fig=False,
                                                    print_data=False)
                super().save_dist(dist, dist_name)
            return dist

Once you've saved this file, you should be able to use the ``trafpy.benchmarker.BenchmarkImporter``
class to import your default benchmark just as you would any default ``TrafPy`` benchmark::

    from trafpy.benchmarker import BenchmarkImporter

    importer = BenchmarkImporter(benchmark_version='v001', load_prev_dists=False)
    net = tpg.gen_fat_tree(k=4)
    dists = importer.get_benchmark_dists(benchmark_name='my_default_benchmark', eps=net.graph['endpoints'], racks_dict=net.graph['rack_to_ep_dict'])

If you wish to share this default benchmark with the community, please feel free
to make a contriubution to the open-source ``TrafPy`` project. See the Contribute
guide for details.
