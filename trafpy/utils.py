def seed_stochastic_modules_globally(numpy_module,
                                     random_module,
                                     default_seed=0,
                                     numpy_seed=None, 
                                     random_seed=None,
                                     ):
    '''Seeds any stochastic modules so get reproducible results.'''
    if numpy_seed is None:
        numpy_seed = default_seed
    if random_seed is None:
        random_seed = default_seed

    numpy_module.random.seed(numpy_seed)
    random_module.seed(random_seed)
