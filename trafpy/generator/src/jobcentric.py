from trafpy.generator.src.dists import val_dists, node_dists
from trafpy.generator.src import tools
from trafpy.generator.src.flowcentric import duplicate_demands_in_demand_data_dict
from trafpy.utils import get_class_from_path

import numpy as np
import networkx as nx
import time
import multiprocessing
import math
import random
import matplotlib.pyplot as plt
import json
from tqdm import tqdm # progress bar




class JobGenerator:
    def __init__(self,
                 eps,
                 node_dist,
                 flow_size_dist,
                 interarrival_time_dist,
                 num_ops_dist,
                 network_load_config,
                 c=1.5,
                 run_time_gaussian_noise_mean=0,
                 run_time_gaussian_noise_sd=0.2,
                 round_op_run_time_to_nearest=0.0001,
                 prob_data_dependency=0.8,
                 use_multiprocessing=True,
                 min_num_demands=6000,
                 max_num_demands=None,
                 jensen_shannon_distance_threshold=0.1,
                 min_last_demand_arrival_time=None,
                 auto_node_dist_correction=False,
                 check_dont_exceed_one_ep_load=True,
                 flow_packer_cls='trafpy.generator.src.packers.flow_packer_v2.FlowPackerV2',
                 flow_packer_kwargs=None,
                 print_data=False,
                 **kwargs):
        '''
        Args:
            network_load_config (dict): Dict of form {'network_rate_capacity': <int/float>, 'target_load_fraction': <float>, 'disable_timeouts': <bool>, 'return_new_interarrival_time_dist': <bool>},
                where network_rate_capacity is the maximum rate (in e.g. Gbps) at which
                information can be reliably transmitted over the communication network
                which the demand data will be inserted into, and where target_load_fraction
                is the fraction of the network rate capacity being requested by the demands
                (e.g. target_load_fraction=0.75 would generate demands which request
                a load that is 75% of the network rate capacity from the first to 
                the last demand arriving). If 'target_load_fraction' is None, won't adjust
                inter arrival time dist at all to meet network load.
            auto_node_dist_correction (bool): Set to True if you want TrafPy to
                automatically make invalid node distributions valid. If True, invalid
                node distributions where more load is being assigned to a end point
                link than the end point link has bandwidth will be changed by 
                removing the invalid end point link load to its maximum 1.0 load
                and distributing the removed load across all other valid links
                uniformly.
            max_num_demands (int): If not None, will not exceed this number of demands,
                which can help if you find you are exceeding memory limitations in
                your simulations. However, this will also mean that the (1) jensen_shannon_distance_threshold
                and (2) min_last_demand_arrival_time you specifiy may not be met. To
                ensure these are met, you must set max_num_demands to None.
            jensen_shannon_distance_threshold (float): Maximum jensen shannon distance
                required of generated random variables w.r.t. discretised dist they're generated from.
                Must be between 0 and 1. Distance of 0 -> distributions are exactly the same.
                Distance of 1 -> distributions are not at all similar.
                https://medium.com/datalab-log/measuring-the-statistical-similarity-between-two-samples-using-jensen-shannon-and-kullback-leibler-8d05af514b15
                N.B. To meet threshold, this function will keep doubling num_demands
            check_dont_exceed_one_ep_load (bool): If True, when packing flows (assigning
                src-dst node pairs according to specified node distribution), will ensure
                that don't exceed 1.0 load on any end points. If this is not possible,
                will raise an Exception. If False, no exception will be raised, but run
                risk of exceeding 1.0 end point load, which for some users might be
                detrimental to their system.

        '''

        self.started = time.time()

        self.eps = eps
        self.node_dist = node_dist
        self.flow_size_dist = flow_size_dist
        self.interarrival_time_dist = interarrival_time_dist
        self.num_ops_dist = num_ops_dist
        self.c = c
        self.max_num_demands = max_num_demands
        if max_num_demands is not None:
            self.num_demands = min(min_num_demands, max_num_demands)
        else:
            self.num_demands = min_num_demands
        self.network_load_config = network_load_config
        self.c = c
        self.prob_data_dependency = prob_data_dependency
        self.run_time_gaussian_noise_mean = run_time_gaussian_noise_mean
        self.run_time_gaussian_noise_sd = run_time_gaussian_noise_sd
        self.round_op_run_time_to_nearest = round_op_run_time_to_nearest
        self.use_multiprocessing = use_multiprocessing
        # self.use_multiprocessing = True
        self.min_last_demand_arrival_time = min_last_demand_arrival_time
        self.auto_node_dist_correction = auto_node_dist_correction
        self.jensen_shannon_distance_threshold = jensen_shannon_distance_threshold
        self.check_dont_exceed_one_ep_load = check_dont_exceed_one_ep_load
        self.flow_packer_cls = flow_packer_cls
        if flow_packer_kwargs is None:
            self.flow_packer_kwargs = {}
        else:
            self.flow_packer_kwargs = flow_packer_kwargs
        self.print_data = print_data

        self.num_nodes, self.num_pairs, self.node_to_index, self.index_to_node = tools.get_network_params(self.eps)

        if self.network_load_config['target_load_fraction'] is not None:
            if self.network_load_config['target_load_fraction'] > 0.95:
                raise Exception('Target load fraction {} is invalid. Must be <= 0.95.'.format(self.network_load_config['target_load_fraction']))

        if not self.check_dont_exceed_one_ep_load:
            print('WARNING: check_dont_exceed_one_ep_load is set to False. This may result in end point loads going above 1.0, which for some users might be detrimental to the systems they want to test.')

    def create_job_centric_demand_data(self, return_packing_time=False, return_packing_jensen_shannon_distance=False):
        '''
        N.B. Currently only applying jensen_shannon_distance_threshold requirement
        to num_ops, not to flow size or interarrival time etc. Do this because
        otherwise may require far too many demands (jobs) -> very memory intensive.
        In future, should try to fix this and apply threshold requirement to all
        distributions.
        '''
        # multiprocessing params
        num_processes = 10
        maxtasksperchild = 1

        # num ops
        num_ops = val_dists.gen_rand_vars_from_discretised_dist(unique_vars=list(self.num_ops_dist.keys()),
                                                                probabilities=list(self.num_ops_dist.values()),
                                                                num_demands=self.num_demands,
                                                                jensen_shannon_distance_threshold=self.jensen_shannon_distance_threshold)
        # update num_demands in case jensen-shannon distance threshold required num_demands to be increased
        self.num_demands = max(len(num_ops), self.num_demands)

        if self.max_num_demands is not None:
            if self.num_demands > self.max_num_demands:
                print('WARNING: max_num_demands is {} but needed {} jobs to meet jensen_shannon_distance_threshold {}. Capping num_demands to max_num_demands, therefore may not meet jensen_shannon_distance_threshold specified. Increase max_num_demands to ensure you meet the jensen_shannon_distance_threshold.'.format(self.max_num_demands, len(num_ops), self.jensen_shannon_distance_threshold))
                self.num_demands = self.max_num_demands
                num_ops = val_dists.gen_rand_vars_from_discretised_dist(unique_vars=list(self.num_ops_dist.keys()),
                                                                        probabilities=list(self.num_ops_dist.values()),
                                                                        num_demands=self.num_demands)

        # init job graphs
        jobs = self._init_job_graphs(num_ops=num_ops,
                                     c=self.c,
                                     prob_data_dependency=self.prob_data_dependency,
                                     use_multiprocessing=self.use_multiprocessing,
                                     print_data=self.print_data)
        
        # flow sizes
        # total num flows == total num data dependencies (some will not become flows if packer sets src == dst)
        # N.B. Might actually work out perfectly without dropped flows if FlowPacker never allocates src==dst?
        total_num_data_deps = 0
        for job in jobs:
            total_num_data_deps += job.graph['num_data_deps']
        flow_sizes = val_dists.gen_rand_vars_from_discretised_dist(unique_vars=list(self.flow_size_dist.keys()),
                                                                   probabilities=list(self.flow_size_dist.values()),
                                                                   num_demands=total_num_data_deps)

        # job interarrival times
        interarrival_times = val_dists.gen_rand_vars_from_discretised_dist(unique_vars=list(self.interarrival_time_dist.keys()),
                                                                           probabilities=list(self.interarrival_time_dist.values()),
                                                                           num_demands=self.num_demands)

        if self.network_load_config['target_load_fraction'] is not None:
            # adjust overall interarrival time dist until overall load <= user-specified load
            interarrival_times = self._adjust_demand_load(flow_sizes,
                                                          interarrival_times)

        # corresponding job event (arrival) times
        event_times = tools.gen_event_times(interarrival_times)
        index, event_times_sorted = np.argsort(event_times), np.sort(event_times)

        # job ids
        job_ids = ['job_'+str(i) for i in range(self.num_demands)]
        establish = [1 for _ in range(self.num_demands)]

        # flow ids
        # flow_ids = ['flow_'+str(i) for i in range(total_num_data_deps)]
        flow_ids = []

        # set which flows will be allocated to which jobs
        self.job_id_to_flow_indices = {}
        last_flow_idx = 0
        for i, job in enumerate(jobs):
            job_id = job_ids[i]
            self.job_id_to_flow_indices[job_id] = [last_flow_idx, last_flow_idx+job.graph['num_data_deps']]
            for f_idx in range(last_flow_idx, last_flow_idx+job.graph['num_data_deps']):
                unique_id = job_id + '_flow_{}'.format(f_idx)
                flow_ids.append(unique_id)
            last_flow_idx += job.graph['num_data_deps']

        # pack the flows into src-dst pairs to meet src-dst pair load config requirements of node_dist
        if isinstance(self.flow_packer_cls, str):
            # load packer class from string path
            flow_packer_cls = get_class_from_path(self.flow_packer_cls)
        else:
            flow_packer_cls = self.flow_packer_cls
        packer = flow_packer_cls(self,
                                 self.eps,
                                 self.node_dist,
                                 flow_ids,
                                 flow_sizes,
                                 interarrival_times,
                                 network_load_config=self.network_load_config,
                                 auto_node_dist_correction=self.auto_node_dist_correction,
                                 check_dont_exceed_one_ep_load=self.check_dont_exceed_one_ep_load,
                                 **self.flow_packer_kwargs,
                                 )
        packer.reset()
        packed_flows = packer.pack_the_flows()
        self.packing_time = packer.packing_time
        self.packing_jensen_shannon_distance = packer.packing_jensen_shannon_distance


        
        # allocate flows to data deps in jobs and allocate attrs to each flow
        # pbar = tqdm(total=len(jobs),
                    # desc='Allocating job flow attrs',
                    # miniters=1, 
                    # # mininterval=1,
                    # # maxinterval=1, # 2
                    # leave=False,
                    # smoothing=0) # 1
        tasks = [] # for multiprocessing
        start = time.time()
        job_idx = 0
        print('Allocating job flow attrs...')
        if self.use_multiprocessing:
            pool = multiprocessing.Pool(processes=num_processes,maxtasksperchild=maxtasksperchild)
            results = [pool.apply_async(self._allocate_job_flow_attrs, args=(job, job_idx, job_ids, packed_flows, None,)) for job, job_idx in zip(jobs, range(len(jobs)))]
            # results = [pool.apply_async(self._allocate_job_flow_attrs, args=(job, job_idx, job_ids, packed_flows, None,), callback=lambda _: pbar.update(1)) for job, job_idx in zip(jobs, range(len(jobs)))]
            pool.close()
            pool.join()
            output = [p.get() for p in results]
            del pool
            jobs = output
        else:
            _jobs = []
            for job in jobs:
                _jobs.append(self._allocate_job_flow_attrs(job, job_idx, job_ids, packed_flows))
                job_idx += 1
            jobs = _jobs
        end = time.time()
        # pbar.close()
        print('Allocated flow attrs for {} jobs in {} seconds.'.format(len(jobs), end-start))

        # set job op run times
        # pbar = tqdm(total=len(jobs),
                # desc='Setting op run times',
                    # miniters=1, 
                    # # mininterval=1,
                    # # maxinterval=1, # 2
                    # leave=False,
                    # smoothing=0) # 1
        print('Setting op run times...')
        start = time.time()
        _jobs = []
        for job in jobs:
            job = self._set_job_op_run_times(job, 
                                           self.run_time_gaussian_noise_mean, 
                                           self.run_time_gaussian_noise_sd, 
                                           self.round_op_run_time_to_nearest)
            # pbar.update(1)
            _jobs.append(job)
        jobs = _jobs
        end = time.time()
        # pbar.close()
        print('Set op run times of {} jobs in {} seconds'.format(len(jobs), end-start))

        _jobs = [jobs[i] for i in index]

        demand_data = {'job_id': np.asarray(job_ids)[index],
                       'job': _jobs,
                       'event_time': event_times_sorted,
                       'establish': np.asarray(establish)[index].astype(int),
                       'index': index}

        # get flowcentric demand data
        demand_data['flow_id'] = flow_ids
        demand_data['sn'] = []
        demand_data['dn'] = []
        demand_data['flow_size'] = []
        for flow in packed_flows.keys():
            demand_data['sn'].append(packed_flows[flow]['src'])
            demand_data['dn'].append(packed_flows[flow]['dst'])
            demand_data['flow_size'].append(packed_flows[flow]['size'])

        if self.min_last_demand_arrival_time is not None:
            # duplicate demands until get duration >= user-specified duration
            adjustment_factor = self.min_last_demand_arrival_time / max(demand_data['event_time'])
            num_duplications = math.ceil(math.log(adjustment_factor, 2))
            if self.max_num_demands is not None:
                if (2**num_duplications) * len(demand_data['job_id']) > self.max_num_demands:
                    print('WARING: max_num_demands is {} but have specified min_last_demand_arrival_time {}. Would need {} demands to reach this min_last_demand_arrival_time, therefore must increase max_num_demands (or set to None) if you want to meet this min_last_demand_arrival_time.'.format(self.max_num_demands, self.min_last_demand_arrival_time, (2**num_duplications)*len(demand_data['job_id'])))
                    return demand_data
            if num_duplications > 0:
                demand_data = duplicate_demands_in_demand_data_dict(demand_data, 
                                                                    num_duplications=num_duplications,
                                                                    use_multiprocessing=True,
                                                                    num_processes=10,
                                                                    maxtasksperchild=1)

        if not return_packing_time and not return_packing_jensen_shannon_distance:
            returns = demand_data
        else:
            returns = [demand_data]
            if return_packing_time:
                returns.append(self.packing_time)
            if return_packing_jensen_shannon_distance:
                returns.append(self.packing_jensen_shannon_distance)
        return returns


    def _set_job_op_run_times(self,
                             job, 
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

        # Assume flow sizes are given in Bytes, but that we want in MB to be able to apply below equation for generation op run times similar to DeepMind paper method https://arxiv.org/pdf/1905.02494.pdf
        # assuming this will give op run times in units of us
        conversion = 1e6

        run_times = []
        for flow in job.edges:
            # set op run times
            parent_op = flow[0]
            parent_op_parent_flows = job.in_edges(parent_op)
            parent_op_child_flows = job.out_edges(parent_op)
            size_input_flows, size_output_flows = 0, 0
            for f in parent_op_parent_flows:
                size_input_flows += int(job.edges[f]['attr_dict']['flow_size'] / conversion)
            for f in parent_op_child_flows:
                size_output_flows += int(job.edges[f]['attr_dict']['flow_size'] / conversion)
            if parent_op == 'source':
                run_time = 0
            else:
                r = np.random.normal(loc=run_time_gaussian_noise_mean,scale=run_time_gaussian_noise_sd)   
                info_size = size_input_flows+size_output_flows
                run_time = (info_size) + (r*(info_size))
                run_times.append(run_time)
            job.edges[flow]['attr_dict']['parent_op_run_time'] = run_time
        
        # set list of op run times as global graph attr
        job.graph['op_run_times'] = run_times
        job.graph['sum_op_run_times'] = sum(run_times)


        if jobs is not None:
            jobs.append(job)
        else:
            return job






    def _allocate_job_flow_attrs(self,
                                 job, 
                                 job_idx, 
                                 job_ids, 
                                 flows, 
                                 jobs=None):
        '''
        If doing multi processing i.e. generating multiple jobs in parallel,
        must give multiprocessing.Manager().list() object as jobs attr of this
        function so that function can append to multiprocessing manager list.
        If not doing multiprocessing, leave attr as jobs=None
        '''
        job_id = job_ids[job_idx]
        job.graph['job_id'] = job_id

        sum_flow_info = 0

        # get flows allocated to this job's data deps
        flow_indices = self.job_id_to_flow_indices[job_id]
        fs = iter(list(flows.keys())[flow_indices[0]:flow_indices[1]])

        # go through edges going out of each op in job DAG
        for op in job.nodes:
            edges = job.out_edges(op)
            parent_dep_edges = [tuple(op) for op in job.in_edges(op)]
            parent_dep_flow_ids = []
            for dep in parent_dep_edges:
                parent_dep_flow_ids.append('flow_'+str(job.edges[dep]['dep_id']))

            for edge in edges:
                # src = job.nodes[flow[0]]['attr_dict']['machine']
                # dst = job.nodes[flow[1]]['attr_dict']['machine']
                if job.edges[edge]['dependency'] == 1:
                    # is a data dependency -> becomes a flow
                    f = next(fs)
                    src, dst = flows[f]['src'], flows[f]['dst']
                else:
                    # is a control dependency, just randomly choose src dst pair
                    src = np.random.choice(self.eps)
                    dst = np.random.choice(self.eps)

                parent_op = edge[0]
                child_op = edge[1]
                child_dep_edges = [tuple(child_op) for child_op in job.out_edges(child_op)]

                child_dep_flow_ids = []
                for dep in child_dep_edges:
                    child_dep_flow_ids.append('flow_'+str(job.edges[dep]['dep_id']))

                if src != dst and job.edges[edge]['dependency'] == 1:
                    # edge in job graph is a data dependency and becomes flow
                    flow_size = flows[f]['size']
                    sum_flow_info += flow_size
                    dependency_type = 'data_dep'
                else:
                    # edge head & tail at same machine or is control dep therefore not a flow
                    # if src == dst, is still a dependency but not a data dependency,
                    # therefore can register this as a control dependency
                    flow_size = 0
                    dependency_type = 'control_dep'

                flow_stats={'sn': src,
                            'dn': dst,
                            'flow_size': float(flow_size),
                            'flow_id': 'flow_'+str(job.edges[edge]['dep_id']),
                            'job_id': job_ids[job_idx],
                            'edge': edge,
                            'parent_dependency_edges': parent_dep_edges,
                            'parent_dependency_flow_ids': parent_dep_flow_ids,
                            'child_dependency_edges': child_dep_edges,
                            'child_dependency_flow_ids': child_dep_flow_ids,
                            'parent_op': parent_op,
                            'child_op': child_op,
                            'dependency_type': dependency_type,
                            'establish': None, # None
                            'event_time': None} # None
                
                job.add_edge(edge[0],edge[1],attr_dict=flow_stats)

        job.graph['sum_flow_info'] = sum_flow_info

        if jobs is not None:
            jobs.append(job)
        else:
            return job













    def _calc_overall_load_rate(self, flow_sizes, interarrival_times):
        '''Returns load rate (info units per unit time).'''
        info_arrived = self._calc_total_info_arrived(flow_sizes)
        first_flow_arrival_time, last_flow_arrival_time = self._get_first_last_flow_arrival_times(interarrival_times)
        duration = last_flow_arrival_time - first_flow_arrival_time
        return info_arrived/duration

    def _calc_total_info_arrived(self, flow_sizes):
        return np.sum(flow_sizes)

    def _get_first_last_flow_arrival_times(self, interarrival_times):
        event_times = tools.gen_event_times(interarrival_times)
        return min(event_times), max(event_times)

    def _change_interarrival_times_by_factor(self, interarrival_times, factor):
        '''Updates self.interarrival_time_dist by a specified factor and returns new interarrival times.'''
        new_interarrival_time_dist = {}
        for rand_var, prob in self.interarrival_time_dist.items():
            new_rand_var = rand_var * factor
            new_interarrival_time_dist[new_rand_var] = prob

        # update interarrival time dist
        self.interarrival_time_dist = new_interarrival_time_dist

        # gen new interarrival times
        interarrival_times *= factor

        return interarrival_times


    def _adjust_demand_load(self,
                            flow_sizes,
                            interarrival_times):
        # total info arriving (sum of flow sizes) is fixed
        # therefore to adjust load, must adjust duration by adjusting interarrival time dist
        load_rate = self._calc_overall_load_rate(flow_sizes, interarrival_times)
        load_fraction = load_rate / self.network_load_config['network_rate_capacity']
        adjustment_factor = load_fraction / self.network_load_config['target_load_fraction'] 
        interarrival_times = self._change_interarrival_times_by_factor(interarrival_times, adjustment_factor)

        return interarrival_times
        

    
    def _init_job_graphs(self,
                         num_ops,
                         c,
                         prob_data_dependency=0.8,
                         use_multiprocessing=True,
                         print_data=False):
        '''
        If doing multi processing i.e. generating multiple jobs in parallel,
        must give multiprocessing.Manager().list() object as jobs attr of this
        function so that function can append to multiprocessing manager list.
        If not doing multiprocessing, leave attr as jobs=None & func will
        return a single job.

        num_ops (list): List of number of operations for each job. Length of list
            is number of jobs to generate.

        '''
        # job_ids = ['job_'+str(idx) for idx in range(num_demands)]
        # jobs = np.array(np.zeros((len(job_ids))),dtype=object)
        num_processes = 10
        num_jobs = len(num_ops)
        tasks = [] # for multiprocessing
        start = time.time()
        if print_data:
            print('Generating {} job computation graphs...'.format(num_jobs))
        if use_multiprocessing:
            jobs = multiprocessing.Manager().list() # for demand generation
            # pool = multiprocessing.Pool(multiprocessing.cpu_count())
            pool = multiprocessing.Pool(num_processes)
            # results = [pool.apply_async(self._init_job_graph, args=(int(val_dists.gen_rand_vars_from_discretised_dist(unique_vars=list(num_ops_dist.keys()),probabilities=list(num_ops_dist.values()),num_demands=1)[0]), c, prob_data_dependency, jobs, True,)) for _ in range(num_jobs)]
            results = [pool.apply_async(self._init_job_graph, args=(int(num_ops[i]), c, prob_data_dependency, jobs, print_data,)) for i in range(num_jobs)]
            pool.close()
            pool.join()
            del pool
        else:
            jobs = []
            for i in range(num_jobs):
                jobs.append(self._init_job_graph(int(num_ops[i]), c, prob_data_dependency))

        end = time.time()
        if print_data:
            print('Generated {} job graphs in {} seconds'.format(num_jobs, end-start))

        return jobs

    def _init_job_graph(self,
                        num_ops,
                        c,
                        prob_data_dependency=0.8,
                        jobs=None,
                        print_data=False):
        '''
        num_ops (int): Number of operations in job graph to generate.
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
        num_data_deps, num_control_deps = 0, 0
        for edge in directed_job.edges:
            dep = np.random.choice([1, 0], p=[prob_data_dependency, 
                                              1-prob_data_dependency])
            directed_job.add_edge(edge[0], edge[1], dep_id=dep_id, dependency=dep)
            if dep == 1:
                num_data_deps += 1
            else:
                num_control_deps += 1
            dep_id+=1
        end = time.time()
        if print_data:
            print('Time to define control and data deps: {}'.format(end-start))

        # get graph diameter
        diameter = tools.calc_graph_diameter(directed_job)

        # set global job attrs
        directed_job.graph['num_data_deps'] = num_data_deps
        directed_job.graph['num_control_deps'] = num_control_deps
        directed_job.graph['graph_diameter'] = diameter

        if jobs is not None:
            # doing multiprocessing, must append to manager list
            jobs.append(directed_job)
        else:
            # not doing multiprocessing, return single job
            return directed_job











def gen_job_event_dict(demand_data, event_iter):
    job = demand_data['job'][event_iter]
    establish = demand_data['establish'][event_iter]
    time_arrived = demand_data['event_time'][event_iter]
    
    flow_stats = {flow: job.get_edge_data(flow[0], flow[1]) for flow in job.edges} 
    for flow in flow_stats:
        flow_stats[flow]['attr_dict']['establish'] = establish
    
    flow_dicts = [tools.gen_event_dict(flow_stats[flow]['attr_dict']) for flow in flow_stats]

    event_dict = {'job_id': demand_data['job_id'][event_iter],
                  'establish': establish,
                  'time_arrived': time_arrived,
                  'time_completed': None,
                  'flow_dicts': flow_dicts}

    return event_dict


def get_job_dependency_stats(job):
    '''
    Get stats of all dependencies in a single job and returns as a list.
    ''' 
    job_dep_stats = []
    for op in job.nodes:
        deps = job.out_edges(op)
        for dep in deps:
            dep_stats = job.get_edge_data(dep[0],dep[1])
            job_dep_stats.append(dep_stats)

    return job_dep_stats 
            

def get_job_demand_data_dependency_stats(demand_data):
    '''
    Gets stats of all dependencies of each job in demand_data. Returns these
    stats as a dict {job_id: dependency_stats}
    '''
    dep_stats = {job_id: [] for job_id in demand_data['job_id']}
    for idx in range(len(demand_data['job_id'])):
        job_id = demand_data['job_id'][idx]
        job = demand_data['job'][idx]
        stats = get_job_dependency_stats(job)
        dep_stats[job_id] = stats

    return dep_stats




def draw_job_graph(job, 
                   node_size=500, 
                   font_size=15, 
                   linewidths=1, 
                   fig_scale=1,
                   draw_labels=True,
                   show_fig=False,
                   directed_graph=True):
    '''
    Draws single job graph
    '''
    pos=nx.nx_agraph.graphviz_layout(job,prog='dot',root='source')
    
    srcsnk = []
    ops = []
    for node in job.nodes:
        if node == 'source' or node == 'sink':
            srcsnk.append(node)
        else:
            ops.append(node)

    if directed_graph:
        data_deps = []
        control_deps = []
        for edge in job.edges:
            if job.edges[edge]['dependency'] == 1:
                # data dependency
                data_deps.append(edge)
            else:
                # control dependency
                control_deps.append(edge)

    fig = plt.figure(figsize=[15*fig_scale,15*fig_scale])
    # nodes
    nx.draw_networkx_nodes(job,
                           pos,
                           nodelist=srcsnk,
                           node_size=node_size,
                           node_color='#47c974',
                           linewidths=linewidths,
                           label='Source/Sink')
    nx.draw_networkx_nodes(job,
                           pos,
                           nodelist=ops,
                           node_size=node_size,
                           node_color='#bd3631',
                           linewidths=linewidths,
                           label='Op')
    # edges
    if directed_graph:
        nx.draw_networkx_edges(job,
                               pos, 
                               edgelist=data_deps,
                               edge_color='#379bbf',
                               width=1,
                               label='Data dependency')
        nx.draw_networkx_edges(job,
                               pos,
                               edgelist=control_deps,
                               edge_color='k',
                               width=1,
                               label='Control dependency')
    else:
        nx.draw_networkx_edges(job,
                               pos,
                               edgelist=job.edges,
                               edge_color='k',
                               width=1,
                               label='Dependency')
    # labels
    if draw_labels:
        nx.draw_networkx_labels(job,
                                pos, 
                                font_size=font_size,
                                font_color='k',
                                font_family='sans-serif',
                                font_weight='normal',
                                alpha=1.0)
    else:
        pass
    
    plt.legend(labelspacing=2)
    if show_fig:
        plt.show()
    plt.close()

    return fig



def draw_job_graphs(demand_data=None,
                    job_graphs=[],
                    node_size=500,
                    font_size=15,
                    linewidths=1,
                    fig_scale=1.25,
                    draw_labels=True,
                    show_fig=False,
                    path_to_save=None):
    '''
    Draws list of specified job graphs. If no job graphs specified,
    plots all job graphs
    '''
    job_graphs = list(job_graphs)
    if len(job_graphs) == 0:
        # plot all job graphs
        job_graphs = demand_data['job']
        assert demand_data is not None, 'must provide job demand data or list of job graphs'

    figs = []
    plotted_jobs = [] # record plotted jobs so dont double plot est==0/1
    for job in job_graphs:
        if job not in plotted_jobs:
            figs.append(draw_job_graph(job,
                                       node_size=node_size,
                                       font_size=font_size,
                                       linewidths=linewidths,
                                       fig_scale=fig_scale,
                                       show_fig=show_fig,
                                       draw_labels=draw_labels))
            plotted_jobs.append(job)
        else:
            # already plotted job
            pass

    # save job graphs
    if path_to_save is not None:
        tools.pickle_data(path_to_save, figs)

    return figs

























