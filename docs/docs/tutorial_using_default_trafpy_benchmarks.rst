Using Default TrafPy Benchmarks
===============================

To load the pre-defined default ``TrafPy`` benchmarks, use the ``trafpy.benchmarker.BenchmarkImporter``
class.

.. nbplot::
    
    >>> from trafpy.benchmarker import BenchmarkImporter

Initialise a network of your choice for which you would like to generate some
benchmark traffic for

.. nbplot::

    >>> import trafpy.generator as tpg
    >>> net = tpg.gen_fat_tree(k=4)

``TrafPy`` is able to generate arbitrary distributions and therefore arbitrary
traffic traces. As the nature of applications handled by real data centres
changes, so too will the resultant traffic traces. Therefore, the initial
benchmarks established for ``TrafPy`` have been stored under
``benchmark_version='v001'`` with the anticipation that future benchmark
versions will be established with new and evolving traffic traces.

``TrafPy`` benchmarks are generated from sets of ``TrafPy`` parameters rather
than stored as data sets of flows. This enables the same benchmark to be used
to generate the same traffic trace characteristics for different networks.
By setting ``load_prev_dists=False``, ``TrafPy`` will re-generate distributions
for your network by sampling from these underlying ``TrafPy`` parameters. If you
want to use the same distributions to generate new data sets to e.g. repeat some tests
sampling from the same benchmark distributions, you should set ``load_prev_dists=True``,
however bare in mind that if the number of end points in your network has changed
since you last generated the benchmark distributions with ``load_prev_dists=False``,
then ``TrafPy`` will raise an error.

Initialise the ``trafpy.benchmarker.BenchmarkImporter``

.. nbplot::
    
    >>> importer = BenchmarkImporter(benchmark_version='v001', load_prev_dists=False)

The ``trafpy.benchmarker.BenchmarkImporter`` can be used to import pre-defined
default ``TrafPy`` benchmarks and generate their corresponding distributions for your
custom network. To see the valid default benchmark names supported by the importer,
run:

.. nbplot::

    >>> print(importer.default_benchmark_names)
    ['commercial_cloud', 'jobcentric_prototyping', 'private_enterprise',
    'rack_sensitivity_0', 'rack_sensitivity_02', 'rack_sensitivity_04',
    'rack_sensitivity_06', 'rack_sensitivity_08', 'skewed_nodes_sensitivity_0',
    'skewed_nodes_sensitivity_005', 'skewed_nodes_sensitivity_01',
    'skewed_nodes_sensitivity_02', 'skewed_nodes_sensitivity_04',
    'social_media_cloud', 'tensorflow', 'uniform', 'university']

Decide which default benchmarks you would like to generate for your system,
and then simply call the ``trafpy.benchmarker.BenchmarkImporter.get_benchmark_dists`` function
to get the node, flow size, and flow interarrival time distributions of this benchmark
adapted to your custom network. For example, we can generate the ``'university'``
benchmark distributions:

.. nbplot::

    >>> dists = importer.get_benchmark_dists(benchmark_name='university', eps=net.graph['endpoints'], racks_dict=net.graph['rack_to_ep_dict'])

The distributions are returned as a dictionary

.. nbplot::
    
    >>> print(dists.keys())
        'node_dist', 'flow_size_dist', 'interarrival_time_dist'

You can now use the ``trafpy.generator.create_demand_data()`` function as usual
to use these benchmark distributions to generate traffic for your own network.
