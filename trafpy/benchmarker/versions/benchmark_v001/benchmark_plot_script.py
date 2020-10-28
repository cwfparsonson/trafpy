from trafpy.benchmarker.versions.benchmark_v001.distribution_generator import DistributionGenerator


if __name__ == '__main__':
    distgen = DistributionGenerator(load_prev_dists=True)
    plots, dists, rand_vars = distgen.plot_benchmark_dists(benchmarks=['uniform', 'university'])

