def get_default_benchmarks():
    '''
    Gets list of default benchmarks in TrafPy.
    '''
    default_benchmarks = ['university',
                          'private_enterprise',
                          'commercial_cloud',
                          'social_media_cloud',
                          'uniform',
                           'skewed_nodes_sensitivity_0', 
                           'skewed_nodes_sensitivity_0.05', 
                           'skewed_nodes_sensitivity_0.1',
                           'skewed_nodes_sensitivity_0.2',
                           'skewed_nodes_sensitivity_0.4',
                           'rack_dist_sensitivity_0',
                           'rack_dist_sensitivity_0.2',
                           'rack_dist_sensitivity_0.4',
                           'rack_dist_sensitivity_0.6',
                           'rack_dist_sensitivity_0.8',
                           'jobcentric_prototyping',
                           'tensorflow']
    return default_benchmarks 
