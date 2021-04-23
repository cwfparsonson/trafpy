from trafpy.generator.src import builder
from trafpy.generator.src.demand import Demand
from trafpy.generator.src.tools import load_data_from_json, unpickle_data

import glob
import json
import time
from sqlitedict import SqliteDict
import os




if __name__ == '__main__':
    '''
    Take path to previously generated json demand data file(s) and organise into
    slots_dict(s), then save as SQLite database in same path.

    '''

    base_path = '/rdata/ong/trafpy/traces/jobcentric/'
    path_to_data = base_path + 'jobcentric_prototyping_k_4_L_2_n_16_chancap1250_numchans1_mldat3.2e5_bidirectional_benchmark_data'
    slot_size = 0.1 # 500.0

    if glob.glob(path_to_data + '/*.json'):
        extension = '.json'
    elif glob.glob(path_to_data + '/*.pickle'):
        extension = '.pickle'
    else:
        raise Exception('Unrecognised or multiple file format in {}'.format(path_to_data))

    _files = sorted(glob.glob(path_to_data + '/*{}'.format(extension)))
    if len(_files) == 0:
        raise Exception('No demand_data files found in {}.'.format(path_to_data))

    for _file in _files:
        if os.path.exists(_file[:-(len(extension))]+'_slotsize_{}_slots_dict.sqlite'.format(slot_size)):
            print('\n{} already has an existing slots_dict database with slot_size {}. Skipping...'.format(_file, slot_size))
        else:
            print('\nGenerating slots_dict database with slot_size {} for {}...'.format(slot_size, _file))
            start = time.time()
            if extension == '.json':
                demand_data = json.loads(load_data_from_json(_file, print_times=False))
            elif extension == '.pickle':
                demand_data = unpickle_data(_file, print_times=False)
            slots_dict = builder.construct_demand_slots_dict(demand_data=demand_data,
                                                            include_empty_slots=True,
                                                            slot_size=slot_size,
                                                            print_info=True)
            with SqliteDict(_file[:-(len(extension))]+'_slotsize_{}_slots_dict.sqlite'.format(slot_size)) as _slots_dict:
                for key, val in slots_dict.items():
                    if type(key) is not str:
                        _slots_dict[json.dumps(key)] = val
                    else:
                        _slots_dict[key] = val
                _slots_dict.commit()
                _slots_dict.close()
            end = time.time()
            print('Generated slots_dict database in {} s.'.format(end-start))

    
    




