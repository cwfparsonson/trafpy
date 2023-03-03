import pathlib
import glob

import torch
import os
import subprocess
import pandas as pd
from io import StringIO

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

def get_class_from_path(path):
    '''
    Path must be the path to the class **without** the .py extension.

    E.g. ddls.module_name.ModuleClass
    '''
    ClassName = path.split('.')[-1]
    path_to_class = '.'.join(path.split('.')[:-1])
    module = __import__(path_to_class, fromlist=[ClassName])
    return getattr(module, ClassName)

def gen_unique_experiment_folder(path_to_save, experiment_name):
    # init highest level folder
    path = path_to_save + '/' + experiment_name + '/'
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    # init folder for this experiment
    path_items = glob.glob(path+'*')
    ids = sorted([int(el.split('_')[-1]) for el in path_items])
    if len(ids) > 0:
        _id = ids[-1] + 1
    else:
        _id = 0
    foldername = f'{experiment_name}_{_id}/'
    pathlib.Path(path+foldername).mkdir(parents=True, exist_ok=False)

    return path + foldername

def get_least_used_gpu():
    '''Returns the GPU index on the current server with the most available memory.'''
    # get devices visible to cuda
    cuda_visible_devices = [int(device) for device in os.environ["CUDA_VISIBLE_DEVICES"].split(',')]

    # get string output of nvidia-smi memory query
    gpu_stats = subprocess.run(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"], stdout=subprocess.PIPE).stdout.decode('utf-8')

    # process query into StringIO object
    gpu_stats_2 = u''.join(gpu_stats)
    gpu_stats_3 = StringIO(gpu_stats_2)
    gpu_stats_3.seek(0)

    # read into dataframe
    gpu_df = pd.read_csv(gpu_stats_3,
                         names=['memory.used', 'memory.free'],
                         skiprows=1)

    # filter any devices not in cuda visible devices
    gpu_df = gpu_df[gpu_df.index.isin(cuda_visible_devices)]

    # get GPU with most free memory
    gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: x.rstrip(' MiB'))
    idx = int(gpu_df['memory.free'].astype(float).idxmax())

    return idx
