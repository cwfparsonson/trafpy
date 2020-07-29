from trafpy.src.dists import val_dists, node_dists
from trafpy.src import tools

import numpy as np
import networkx as nx
import time
import multiprocessing
import math
import random


def set_job_op_run_times(job, 
                         run_time_gaussian_noise_mean, 
                         run_time_gaussian_noise_sd,
                         round_op_run_time_to_nearest,
                         jobs=None):
    '''
    If doing multi processing i.e. generating multiple jobs in parallel,
    must give multiprocessing.Manager().list() object as jobs attr of this
    function so that function can append to multiprocessing manager list.
    If not doing multiprocessing, leave attr as jobs=None
    '''
    for flow in job.edges:
        # set op run times
        parent_op = flow[0]
        parent_op_parent_flows = job.in_edges(parent_op)
        parent_op_child_flows = job.out_edges(parent_op)
        size_input_flows, size_output_flows = 0, 0
        for f in parent_op_parent_flows:
            size_input_flows += job.edges[f]['attr_dict']['flow_size']
        for f in parent_op_child_flows:
            size_output_flows += job.edges[f]['attr_dict']['flow_size']
        if parent_op == 'source':
            run_time = 0
        else:
            r = np.random.normal(loc=run_time_gaussian_noise_mean,scale=run_time_gaussian_noise_sd)   
            info_size = size_input_flows+size_output_flows
            run_time = (info_size) + (r*(info_size))
        job.edges[flow]['attr_dict']['parent_op_run_time'] = run_time
    
    if jobs is not None:
        jobs.append(job)
    else:
        return job


def allocate_job_flow_attrs(job, 
                            job_idx, 
                            job_ids, 
                            poss_flow_sizes, 
                            flow_size_dist, 
                            jobs=None):
    '''
    If doing multi processing i.e. generating multiple jobs in parallel,
    must give multiprocessing.Manager().list() object as jobs attr of this
    function so that function can append to multiprocessing manager list.
    If not doing multiprocessing, leave attr as jobs=None
    '''
    for op in job.nodes:
        flows = job.out_edges(op)
        parent_dep_edges = [tuple(op) for op in job.in_edges(op)]
        parent_dep_flow_ids = []
        for dep in parent_dep_edges:
            parent_dep_flow_ids.append('flow_'+str(job.edges[dep]['dep_id']))

        for flow in flows:
            src = job.nodes[flow[0]]['attr_dict']['machine']
            dst = job.nodes[flow[1]]['attr_dict']['machine']
           
            parent_op = flow[0]
            child_op = flow[1]
            child_dep_edges = [tuple(child_op) for child_op in job.out_edges(child_op)]

            child_dep_flow_ids = []
            for dep in child_dep_edges:
                child_dep_flow_ids.append('flow_'+str(job.edges[dep]['dep_id']))

            
            if src != dst and job.edges[flow]['dependency'] == 1:
                # edge in job graph is a data dependency and becomes flow
                flow_size = np.random.choice(a = list(flow_size_dist.keys()), 
                                             size = None,
                                             p = list(flow_size_dist.values()))
            else:
                # edge head & tail at same machine or is control dep therefore not a flow
                # if src == dst, is still a dependency but not a data dependency,
                # therefore can register this as a control dependency
                flow_size=0
            
            flow_stats={'sn': tuple(list(src))[0],
                        'dn': tuple(list(dst))[0],
                        'flow_size': float(flow_size),
                        'flow_id': 'flow_'+str(job.edges[flow]['dep_id']),
                        'job_id': job_ids[job_idx],
                        'edge': flow,
                        'parent_dependency_edges': parent_dep_edges,
                        'parent_dependency_flow_ids': parent_dep_flow_ids,
                        'child_dependency_edges': child_dep_edges,
                        'child_dependency_flow_ids': child_dep_flow_ids,
                        'parent_op': parent_op,
                        'child_op': child_op,
                        'establish': None, # None
                        'event_time': None} # None
            

            job.add_edge(flow[0],flow[1],attr_dict=flow_stats)

    if jobs is not None:
        jobs.append(job)
    else:
        return job


def allocate_job_ops_to_machines(job, eps, node_dist, jobs=None):
    '''
    If doing multi processing i.e. generating multiple jobs in parallel,
    must give multiprocessing.Manager().list() object as jobs attr of this
    function so that function can append to multiprocessing manager list.
    If not doing multiprocessing, leave attr as jobs=None
    '''
    for op in job.nodes:
        machine = node_dists.gen_demand_nodes(eps=eps,
                                              node_dist=node_dist, 
                                              size=1, 
                                              axis=0)
        op_stats = {'machine': machine}
        job.add_node(op, attr_dict=op_stats)
    
    if jobs is not None:
        jobs.append(job)
    else:
        return job


def gen_job_graph(num_ops, 
                  c,
                  prob_data_dependency=0.8,
                  jobs=None,
                  print_data=False):
    '''
    If doing multi processing i.e. generating multiple jobs in parallel,
    must give multiprocessing.Manager().list() object as jobs attr of this
    function so that function can append to multiprocessing manager list.
    If not doing multiprocessing, leave attr as jobs=None & func will
    return a single job.
    '''
    
    prob_edge = c * (math.log(num_ops)/num_ops)

    threshold = math.sqrt(num_ops) * math.sqrt((math.log(num_ops)/num_ops))
    if prob_edge >= threshold:
        print('WARNING: Threshold for graph_diameter<=2 for n={} graph is \
                {}, but your edge formation prob is {}. Consider lowering \
                prob_edge to < {} to avoid low graph diameters (see \
                https://www.cs.cmu.edu/~avrim/598/chap4only.pdf for more info)'.format(num_ops,threshold,prob_edge,threshold))

    # init undirected graph 0.05
    undirected_job = nx.erdos_renyi_graph(n=num_ops,p=prob_edge,directed=False)
    nx.set_edge_attributes(undirected_job, 0, 'assigned_direction')
    
    # randomly define order of ops to make directed acyclic graph (DAG)
    start = time.time()
    undirected_nodes = list(undirected_job.nodes)
    undirected_edges = list(undirected_job.edges)
    random.shuffle(undirected_nodes)
    ops = undirected_nodes
    directed_job = nx.DiGraph()
    nodes_to_add = {node: 1 for node in undirected_nodes}
    for idx in range(len(undirected_nodes)):
        tail_op = ops[idx]
        flows = undirected_job.edges(tail_op)
        for flow in flows:
            if undirected_job[flow[0]][flow[1]]['assigned_direction'] == 0:
                # flow/edge not yet assigned a direction
                directed_job.add_edge('op_'+str(flow[0]),'op_'+str(flow[1]))
                undirected_job[flow[0]][flow[1]]['assigned_direction'] = 1
                # record that added this node
                nodes_to_add[flow[0]] = 0
            else:
                # edge already assigned a direction by prev op
                pass

    end = time.time()
    if print_data:
        print('Time to define DAG order: {}'.format(end-start))
    
    # if any nodes left to add, add them
    start = time.time()
    for op in nodes_to_add.keys():
        if nodes_to_add[op] == 1:
            directed_job.add_node('op_'+str(op))
    num_nodes = len(list(directed_job.nodes))
    assert num_nodes == num_ops, \
            'ERROR: DAG has {} nodes, but {} ops were specified'.format(num_nodes,num_ops)
    end = time.time()
    if print_data:
        print('Time to add left over node: {}'.format(end-start))

    # connect source & sink nodes to tailess & headless ops respectively
    start = time.time()
    heads_no_in_edges = []
    tails_no_out_edges = []
    for op in directed_job.nodes:
        if len(directed_job.in_edges(op)) == 0:
            heads_no_in_edges.append(op)
        else:
            pass
        if len(directed_job.out_edges(op)) == 0:
            tails_no_out_edges.append(op)
    directed_job.add_node('source')
    directed_job.add_node('sink')
    for op in heads_no_in_edges:
        directed_job.add_edge('source', op)
    for op in tails_no_out_edges:
        directed_job.add_edge(op, 'sink')
    end = time.time()
    if print_data:
        print('Time to connect source and sink nodes: {}'.format(end-start))

    # define control and data dependency edges
    start = time.time()
    dep_id = 0
    for edge in directed_job.edges:
        dep = np.random.choice([1, 0], p=[prob_data_dependency, 
                                          1-prob_data_dependency])
        directed_job.add_edge(edge[0], edge[1], dep_id=dep_id, dependency=dep)
        dep_id+=1
    end = time.time()
    if print_data:
        print('Time to define control and data deps: {}'.format(end-start))

    if jobs is not None:
        # doing multiprocessing, must append to manager list
        jobs.append(directed_job)
    else:
        # not doing multiprocessing, return single job
        return directed_job


def gen_job_graphs(num_jobs, 
                   num_ops_dist, 
                   c=1.5, # 0.375
                   prob_data_dependency=0.8,
                   use_multiprocessing=True,
                   print_data=False):

    tasks = [] # for multiprocessing
    start = time.time()
    if print_data:
        print('Generating {} job computation graphs...'.format(num_jobs))
    if use_multiprocessing:
        jobs = multiprocessing.Manager().list() # for demand generation
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        results = [pool.apply_async(gen_job_graph, args=(int(val_dists.gen_rand_vars_from_discretised_dist(unique_vars=list(num_ops_dist.keys()),probabilities=list(num_ops_dist.values()),num_demands=1)[0]), c, prob_data_dependency, jobs, True,)) for _ in range(num_jobs)]
        pool.close()
        pool.join()
        del pool
    else:
        jobs = []
        for _ in range(num_jobs):
            num_ops = val_dists.gen_rand_vars_from_discretised_dist(unique_vars=list(num_ops_dist.keys()),
                                                               probabilities=list(num_ops_dist.values()),
                                                               num_demands=1)
            jobs.append(gen_job_graph(int(num_ops[0]), c, prob_data_dependency))

    end = time.time()
    if print_data:
        print('Generated {} job graphs in {} seconds'.format(num_jobs, end-start))

    return jobs


def create_job_centric_demand_data(num_demands,
                                   eps,
                                   node_dist, 
                                   flow_size_dist,
                                   interarrival_time_dist,
                                   num_ops_dist,
                                   duration_time_dist=None,
                                   c=0.8,
                                   run_time_gaussian_noise_mean=0,
                                   run_time_gaussian_noise_sd=0.1,
                                   round_op_run_time_to_nearest=0.00001,
                                   use_multiprocessing=True,
                                   print_data=False):
    num_processes=10
    maxtasksperchild=1

    started = time.time()

    # init job graphs
    job_ids = ['job_'+str(idx) for idx in range(num_demands)]
    jobs = np.array(np.zeros((len(job_ids))),dtype=object)
    jobs[:num_demands] = gen_job_graphs(num_jobs=num_demands,
                                        num_ops_dist=num_ops_dist,
                                        c=c,
                                        use_multiprocessing=use_multiprocessing,
                                        print_data=print_data)
    if duration_time_dist is not None:
        for idx in range(num_demands):
            # duplicate
            job_ids.append('job_'+str(idx))
        jobs[num_demands:] = jobs[:num_demands]
        establish = np.concatenate((np.ones((num_demands)), 
                                    np.zeros((num_demands))))
    else:
        establish = np.ones((num_demands))

    poss_flow_sizes = np.arange(min(flow_size_dist.keys()), max(flow_size_dist.keys())+1)
    
    # use node dist to allocate each op to machine
    if print_data:
        print('Allocating each op to a machine...')
    start = time.time()
    for job in jobs[:num_demands]:
        allocate_job_ops_to_machines(job, eps, node_dist)
    end = time.time()
    if print_data:
        print('Allocated ops of all jobs to a machine in {} seconds'.format(end-start))
    
    # allocate attrs to each flow
    if print_data:
        print('Allocating attrs to each flow in job...')
    tasks = [] # for multiprocessing
    start = time.time()
    job_idx = 0
    if use_multiprocessing:
        pool = multiprocessing.Pool(processes=num_processes,maxtasksperchild=maxtasksperchild)
        results = [pool.apply_async(allocate_job_flow_attrs, args=(job, job_idx, job_ids, poss_flow_sizes, flow_size_dist, None,)) for job, job_idx in zip(jobs[:num_demands],range(len(jobs[:num_demands])))]
        pool.close()
        pool.join()
        output = [p.get() for p in results]
        del pool
        jobs[:num_demands] = output
    else:
        for job in jobs[:num_demands]:
            allocate_job_flow_attrs(job, job_idx, job_ids, poss_flow_sizes, flow_size_dist)
            job_idx += 1
    end = time.time()
    if print_data:
        print('Allocated flow attrs of all jobs in {} seconds'.format(end-start))
    
    
    # set op run times
    if print_data:
        print('Setting op run times...')
    start = time.time()
    for job in jobs[:num_demands]:
        set_job_op_run_times(job, run_time_gaussian_noise_mean, run_time_gaussian_noise_sd, round_op_run_time_to_nearest)
    end = time.time()
    if print_data:
        print('Set op run times of all jobs in {} seconds'.format(end-start))

    if duration_time_dist is not None:
        # duplicate
        jobs[num_demands:] = jobs[:num_demands]
    
    ended = time.time()
    if print_data:
        print('Total time to generate & assign attrs of {} jobs: {}'.format(num_demands, ended-started))

    # create event time array
    interarrival_times = val_dists.gen_rand_vars_from_discretised_dist(unique_vars=list(interarrival_time_dist.keys()),
                                                                       probabilities=list(interarrival_time_dist.values()),
                                                                       num_demands=num_demands)
    if duration_time_dist is not None:
        interarrival_times = val_dists.gen_rand_vars_from_discretised_dist(unique_vars=list(interarrival_time_dist.keys()),
                                                                           probabilities=list(interarrival_time_dist.values()),
                                                                           num_demands=num_demands)
    else:
        duration_times = None
    event_times = tools.gen_event_times(interarrival_times, duration_times)
    index, event_times_sorted = np.argsort(event_times), np.sort(event_times)
    
    demand_data = {'job_id': np.asarray(job_ids)[index],
                   'job': jobs[index],
                   'event_time': event_times_sorted,
                   'establish': establish[index].astype(int),
                   'index': index}
    
    
    return demand_data






























