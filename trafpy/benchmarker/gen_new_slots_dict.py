from trafpy.generator.src import builder
from trafpy.generator.src.demand import Demand
from trafpy.generator.src.tools import load_data_from_json

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

    base_path = '/scratch/datasets/trafpy/traces/flowcentric/'
    path_to_json_data = base_path + 'rack_dist_sensitivity_0.2_k_4_L_2_n_16_chancap1250_numchans1_mldat3.2e5_bidirectional_benchmark_data'
    slot_size = 1000.0

    json_files = sorted(glob.glob(path_to_json_data + '/*.json'))
    if len(json_files) == 0:
        raise Exception('No json demand_data files found in {}.'.format(path_to_json_data))
    for json_file in json_files:
        if os.path.exists(json_file[:-5]+'_slotsize_{}_slots_dict.sqlite'.format(slot_size)):
            print('\n{} already has an existing slots_dict database with slot_size {}. Skipping...'.format(json_file, slot_size))
        else:
            print('\nGenerating slots_dict database with slot_size {} for {}...'.format(slot_size, json_file))
            start = time.time()
            demand_data = json.loads(load_data_from_json(json_file, print_times=False))
            slots_dict = builder.construct_demand_slots_dict(demand_data=demand_data,
                                                            include_empty_slots=True,
                                                            slot_size=slot_size,
                                                            print_info=True)
            with SqliteDict(json_file[:-5]+'_slotsize_{}_slots_dict.sqlite'.format(slot_size)) as _slots_dict:
                for key, val in slots_dict.items():
                    if type(key) is not str:
                        _slots_dict[json.dumps(key)] = val
                    else:
                        _slots_dict[key] = val
                _slots_dict.commit()
                _slots_dict.close()
            end = time.time()
            print('Generated slots_dict database in {} s.'.format(end-start))

    
    




