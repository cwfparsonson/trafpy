import numpy as np
import pandas as pd
# from trafpy.generator.src.demand import *
import json
import jsonpickle
import pickle
import bz2
import time
import multiprocessing
import networkx as nx
from networkx.readwrite import json_graph
import os
import scipy




def compute_jensen_shannon_distance(p, q):
    # conv to numpy
    p, q = np.asarray(p), np.asarray(q)

    # calc m
    m = (p + q) / 2

    # compute jensen-shannon divergence
    divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2

    # compute jensen-shannon distance
    distance = np.sqrt(divergence)

    return distance







def get_network_params(eps, all_combinations=False):
    '''Returns basic params of network.

    If all_combinations, will consider all possible pair combinations (i.e. src-dst and dst-src).
    If False, will consider src-dst==dst-src -> get half number of node pairs returned.

    '''
    num_nodes = len(eps)
    if all_combinations:
        # assuming src-dst not equivalent to dst-src
        num_pairs = np.int(((num_nodes**2) - num_nodes))
    else:
        # assuming src-dst==dst-src
        num_pairs = np.int(((num_nodes**2) - num_nodes)/2)
    node_indices = [index for index in range(num_nodes)]
    iterables = zip(eps, node_indices)
    node_to_index = {node: index for node, index in iterables}
    iterables = zip(node_indices, eps)
    index_to_node = {index: node for index, node in iterables}
   
    return num_nodes, num_pairs, node_to_index, index_to_node


def gen_event_times(interarrival_times, 
                    duration_times=None,
                    path_to_save=None):
    '''Use event interarrival times to generate event times.'''
    if duration_times is None:
        event_times = np.zeros((int(len(interarrival_times))))
    else:
        event_times = np.zeros((int(len(interarrival_times))*2))

    # points in time at which establishments occur
    for i in np.arange(1, int(len(interarrival_times))):
        event_times[i] = event_times[i-1] + interarrival_times[i-1]

    if duration_times is not None:
        # points in time at which take downs occur
        for i in np.arange(int(len(interarrival_times)), 2*int(len(interarrival_times))):
            event_times[i] = event_times[i-int(len(interarrival_times))] + duration_times[i-(int(len(interarrival_times))+1)]
    else:
        # only consider arrival times, dont need take downs
        pass
    
    if path_to_save is not None:
        pickle_data(path_to_save, event_times)

    return event_times






def gen_event_dict(demand_data, event_iter=None):
    '''Use demand data dict to generate dict for each event in demand data.'''
    if 'job_id' in demand_data:
        job_centric=True
    else:
        job_centric=False

    if event_iter is not None:
        # need to index demand_data
        size = demand_data['flow_size'][event_iter]
        if demand_data['establish'][event_iter] == 0:
            size *= -1
        sn = demand_data['sn'][event_iter]
        dn = demand_data['dn'][event_iter]
        arrived = demand_data['event_time'][event_iter]
        establish = demand_data['establish'][event_iter]
        if job_centric:
            flow_id = demand_data['flow_id'][event_iter]
            job_id = demand_data['job_id'][event_iter]
            unique_id = job_id+'_'+flow_id
            parent_deps = demand_data['parent_dependency_flow_ids'][event_iter]
            child_deps = demand_data['child_dependency_flow_ids'][event_iter]
            parent_op_run_time = demand_data['parent_op_run_time'][event_iter]
            child_op = demand_data['child_op'][event_iter]
            parent_op = demand_data['parent_op'][event_iter]
            dependency_type = demand_data['dependency_type'][event_iter]
            if len(parent_deps) == 0:
                # no parent dependencies therefore can schedule immediately
                can_schedule=1
            else:
                can_schedule=0
        else:
            flow_id = demand_data['flow_id'][event_iter]
            job_id = None
            unique_id = flow_id
            parent_deps = None
            child_deps = None
            parent_op_run_time = None
            child_op = None
            parent_op = None
            dependency_type = None
            can_schedule = 1 # no dependencies in flow centric
    else:
        # no need to index demand_data
        size = demand_data['flow_size']
        if demand_data['establish'] == 0:
            size *= -1
        sn = demand_data['sn']
        dn = demand_data['dn']
        arrived = demand_data['event_time']
        establish = demand_data['establish']
        if job_centric:
            # flows in jobs have unique ids & dependencies
            flow_id = demand_data['flow_id']
            job_id = demand_data['job_id']
            unique_id = job_id+'_'+flow_id
            parent_deps = demand_data['parent_dependency_flow_ids']
            child_deps = demand_data['child_dependency_flow_ids']
            parent_op_run_time = demand_data['parent_op_run_time']
            child_op = demand_data['child_op']
            parent_op = demand_data['parent_op']
            dependency_type = demand_data['dependency_type']
            if len(parent_deps) == 0:
                # no parent dependencies therefore can schedule immediately
                can_schedule=1
            else:
                can_schedule=0
        else:
            flow_id = demand_data['flow_id']
            job_id = None
            unique_id = flow_id
            parent_deps = None
            child_deps = None
            parent_op_run_time = None
            child_op = None
            parent_op = None
            dependency_type = None
            can_schedule=1 # no dependencies in flow centric
        
        
    event_dict = {'flow_id': flow_id,
                  'unique_id': unique_id,
                  'size': size,
                  'src': sn,
                  'dst': dn,
                  'establish': establish,
                  'parent_deps': parent_deps,
                  'completed_parent_deps': [],
                  'child_deps': child_deps,
                  'parent_op_run_time': parent_op_run_time,
                  'time_parent_op_started': None,
                  'parent_op': parent_op,
                  'dependency_type': dependency_type,
                  'child_op': child_op,
                  'can_schedule': can_schedule,
                  'job_id': job_id,
                  'path': None,
                  'channel': None,
                  'packets': None,
                  'packet_size': None,
                  'packets_this_slot': 0,
                  'time_arrived': arrived,
                  'time_completed': None,
                  'k_shortest_paths': None}

    return event_dict

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_data_as_json(path_to_save,
                      data,
                      overwrite=False,
                      print_times=True):
    start = time.time()
    if path_to_save[-5:] != '.json':
        append_json = True
        filename = path_to_save + '.json'
    else:
        append_json = False
        filename = path_to_save
    if overwrite:
        # overwrite prev saved file
        pass
    else:
        # avoid overwriting
        v = 2
        while os.path.exists(str(filename)):
            if append_json:
                filename = path_to_save+'_v{}'.format(v)+'.json'
            else:
                filename = path_to_save[:-5]+'_v{}'.format(v)+'.json'
            v += 1

    dumped = json.dumps(data, cls=NumpyEncoder)
    with open(filename, 'w') as f:
        json.dump(dumped, f)
    # except TypeError:
        # with open(filename, 'w') as f:
            # json.dump(data, f)
        # # # need special jsonpickle serialisation for networkx digraphs
        # # with open(filename, 'w+') as f:
            # # f.write(jsonpickle.encode(json_graph.adjacency_data(data)))

    end = time.time()
    if print_times:
        print('Time to save data to {}: {} s'.format(filename, end-start))

def load_data_from_json(path_to_load,
                        print_times=True):
    start = time.time()
    if path_to_load[-5:] != '.json':
        filename = path_to_load+'.json'
    else:
        filename = path_to_load
    with open(filename) as f:
        data = json.loads(f.read())

    end = time.time()
    if print_times:
        print('Time to load data from {}: {} s'.format(filename,end-start))

    return data



def save_data_as_csv(path_to_save,
                     data,
                     overwrite=False,
                     print_times=True):
    '''Saves data given as a csv file.'''
    start = time.time()
    if path_to_save[-4:] != '.csv':
        append_csv = True
        filename = path_to_save + '.csv'
    else:
        append_csv = False
        filename = path_to_save
    if overwrite:
        # overwrite prev saved file
        pass
    else:
        # avoid overwriting
        v = 2
        while os.path.exists(str(filename)):
            if append_csv:
                filename = path_to_save+'_v{}'.format(v)+'.csv'
            else:
                filename = path_to_save[:-4]+'_v{}'.format(v)+'.csv'
            v += 1
   
    if type(data) == dict:
        try:
            df = pd.DataFrame(data)
        except ValueError:
            # dict values are scalars
            df = pd.DataFrame(data, index=[0])
    
    if type(data) == dict:
        df.to_csv(filename)
    else:
        try:
            np.savetxt(filename, data, delimiter=',')
        except TypeError:
            np.savetxt(filename, data, delimiter=',', fmt='%s')
    
    end = time.time()
    if print_times: 
        print('Time to save data to {}: {} s'.format(filename, end-start))



def pickle_data(path_to_save,
                data,
                overwrite=False, 
                zip_data=True, 
                print_times=True):
    '''Save data as a pickle.'''
    start = time.time()
    if path_to_save[-7:] != '.pickle':
        append_pickle = True
        filename = path_to_save + '.pickle'
    else:
        append_pickle = False
        filename = path_to_save
    if overwrite:
        # overwrite prev saved file
        pass
    else:
        # avoid overwriting
        v = 2
        while os.path.exists(str(filename)):
            if append_pickle:
                filename = path_to_save+'_v{}'.format(v)+'.pickle'
            else:
                filename = path_to_save[:-7]+'_v{}'.format(v)+'.pickle'
            v += 1
    if zip_data:
        filehandler = bz2.open(filename, 'wb')
    else:
        filehandler = open(filename, 'wb')
    pickle.dump(data, filehandler)
    filehandler.close()
    end = time.time()
    if print_times:
        print('Time to save data to {}: {} s'.format(filename,end-start))


def unpickle_data(path_to_load,
                  zip_data=True,
                  print_times=True):
    '''Re-load previously pickled data.'''
    start = time.time()
    if path_to_load[-7:] != '.pickle':
        filename = path_to_load+'.pickle'
    else:
        filename = path_to_load
    if zip_data:
        filehandler = bz2.open(filename, 'rb')
    else:
        filehandler = open(filename, 'rb')
    demand_data = pickle.load(filehandler)
    end = time.time()
    if print_times:
        print('Time to load data from {}: {} s'.format(filename,end-start))

    return demand_data


def calc_graph_diameter(graph):
    '''Calculate diameter of a single graph.'''
    diameter = nx.algorithms.distance_measures.extrema_bounding(to_undirected_graph(graph), compute='diameter')
    return diameter

def calc_graph_diameters(graphs, multiprocessing_type='none', print_times=False):
    '''Calculate diameters of a list of graphs.'''
    start = time.time()
    if multiprocessing_type=='pool':
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        results = [pool.apply_async(calc_graph_diameter, args=(graph,)) for graph in graphs]
        pool.close()
        pool.join()
        diameters = [p.get() for p in results]
    elif multiprocessing_type=='none':
        diameters = [calc_graph_diameter(graph) for graph in graphs]
    end=time.time()
    if print_times:
        print('Time to calc diameters of {} graphs: {}'.format(len(graphs), end-start))

    return diameters
    
def to_undirected_graph(directed_graph):
    '''Converts directed graph to an undirected graph.'''
    edges = directed_graph.edges()
    nodes = directed_graph.nodes()
    
    undirected_graph = nx.Graph()
    for node in nodes:
        undirected_graph.add_node(node)
    for edge in edges:
        undirected_graph.add_edge(edge[0], edge[1])

    return undirected_graph
