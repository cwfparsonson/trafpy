import numpy as np


class EnvAnalyser:

    def __init__(self, env, subject_class_name=None):
        '''
        envs (obj): Environment/simulation object to analyse.
        subject_class_name (str): Name of test subject class. Is useful for when come
            to e.g. plotting multiple envs using EnvsPlotter and want to group
            analysers into classes/subject names being tested (e.g. test subject 
            'scheduler_1' vs. test subject 'scheduler_2') across an
            arbitrary number of tests (e.g. 10 different network loads).
        '''
        self.env = env
        if subject_class_name is None:
            self.subject_class_name = env.sim_name
        else:
            self.subject_class_name = subject_class_name
        self.computed_metrics = False


    def compute_metrics(self, measurement_start_time=None, measurement_end_time=None, bidirectional_links=True, print_summary=False):
        '''
        measurement_start_time (int, float): Simulation time at which to begin recording
            metrics etc.; is the warm-up time
        measurement_end_time (int, float): Simulation time at which to stop recording
            metrics etc.; is the cool-down time
        bidirectional_links (bool): If True, 1 flow occupies bandwidth of 2 endpoint
            links therefore is requesting load of 2*flow_size

        '''
        self.computed_metrics = True
        self.measurement_start_time = measurement_start_time 
        self.measurement_end_time = measurement_end_time 
        self.bidirectional_links = bidirectional_links

        self._compute_flow_summary()
        self._compute_general_summary()
        if self.env.demand.job_centric:
            self._compute_job_summary()

        if print_summary:
            print('\n\n-=-=-=-=-=-=--= Summary -=-=-=-=-=-=-=-')
            self._print_general_summary()
            self._print_flow_summary()
            if self.env.demand.job_centric:
                self._print_job_summary()


    ####################################### GENERAL ##########################################
    def _print_general_summary(self):
        print('\n ~* General Information *~')
        print('Simulation name: \'{}\''.format(self.env.sim_name))
        print('Measurement duration: {} (Start time : {} | End time: {})'.format(self.measurement_duration, self.measurement_start_time, self.measurement_end_time))
        print('Total number of generated demands (jobs or flows) passed to env: {}'.format(self.env.demand.num_demands)) 
        print('Total number of these demands which arrived during measurement period: {}'.format(len(self.arrived_flow_dicts)))
        print('Total info arrived: {} info unit demands arrived'.format(self.total_info_arrived))
        print('Total info transported: {} info unit demands transported'.format(self.total_info_transported))
        print('Load (abs): {} info unit demands arrived per unit time (from first to last flow arriving)'.format(self.load_abs))
        print('Load (frac): {} fraction of network capacity requested (from first to last flow arriving)'.format(self.load_frac))
        print('Throughput (abs): {} info units transported per unit time'.format(self.throughput_abs))
        print('Throughput (frac): {} fraction of arrived info successfully transported'.format(self.throughput_frac))

    def _compute_general_summary(self):
        self.total_info_arrived = self._calc_total_info_arrived()
        self.total_info_transported = self._calc_total_info_transported()
        self.load_abs = self._calc_network_load_abs()
        self.load_frac = self._calc_network_load_frac()
        self.throughput_abs = self._calc_throughput_abs()
        self.throughput_frac = self._calc_throughput_frac()

    def _calc_total_info_arrived(self):
        return sum([self.arrived_flow_dicts[i]['size'] for i in range(len(self.arrived_flow_dicts))])

    def _calc_total_info_transported(self):
        return sum([self.completed_flow_dicts[i]['size'] for i in range(len(self.completed_flow_dicts))])
    
    def _calc_network_load_abs(self):
        '''Calc absolute network load (i.e. is load rate during measurement period).'''
        if self.bidirectional_links:
            # 1 flow occupies 2 links therefore has double load
            return 2*self.total_info_arrived / self.measurement_duration
        else:
            return self.total_info_arrived / self.measurement_duration

    def _calc_network_load_frac(self):
        '''Calc fraction network load (i.e. is fraction of network capacity requested during measurement period).'''
        return self.load_abs / self.env.network.graph['max_nw_capacity']

    def _calc_throughput_abs(self):
        return self.total_info_transported / self.measurement_duration

    def _calc_throughput_frac(self):
        return self.total_info_transported / self.total_info_arrived



    ################################## FLOW ################################################
    def _print_flow_summary(self):
        print('\n ~* Flow Information *~')
        print('Total number of generated flows passed to env (src != dst, dependency_type == \'data_dep\'): {}'.format(self.env.demand.num_flows))
        print('Total number of these flows which arrived during measurement period: {}'.format(len(self.arrived_flow_dicts)))
        print('Time first flow arrived: {}'.format(min(self.flow_times_arrived)))
        print('Time last flow arrived: {}'.format(max(self.flow_times_arrived)))
        print('Total number of flows that were completed: {}'.format(len(self.completed_flow_dicts)))
        print('Total number of flows that were dropped: {}'.format(len(self.dropped_flow_dicts)))
        print('Fraction of arrived flows dropped: {}'.format(self.dropped_flow_frac))
        print('Total number of flows that were left in queue at end of measurement period: {}'.format(len(self.queued_flow_dicts)))
        print('Average flow completion time (FCT): {}'.format(self.average_fct))
        print('99th percentile FCT: {}'.format(self.nn_fct))

    def _compute_flow_summary(self):
        self._compute_flow_arrival_metrics()
        self._compute_flow_completion_metrics()
        self._compute_flow_dropped_metrics()
        self._compute_flow_queued_metrics()
        if self.env.track_grid_slot_evolution:
            self._generate_grid_demands_numpy_array()

    def _generate_grid_demands_numpy_array(self):
        # collect ep link channel demand info into numpy array grid
        self.grid_demands = []
        for ep in self.env.grid_slot_dict.keys():
            for channel in self.env.grid_slot_dict[ep].keys():
                self.grid_demands.append(self.env.grid_slot_dict[ep][channel]['demands'])
        self.grid_demands = np.array([np.array(xi) for xi in self.grid_demands]) # unpack and conv to numpy array

        # conv grid demands to unique integer ids (for colour coding)
        unique_id_counter = 0
        demand_to_id = {}
        for demand_idx in range(self.grid_demands.shape[0]):
            for time_idx in range(self.grid_demands.shape[1]):
                d = self.grid_demands[demand_idx][time_idx]
                if d not in demand_to_id.keys():
                    # not yet encountered demand id, update demand_to_id dict
                    demand_to_id[d] = unique_id_counter
                    unique_id_counter += 1
                    # update grid_demands
                    self.grid_demands[demand_idx][time_idx] = demand_to_id[d]
                else:
                    # update grid_demands
                    self.grid_demands[demand_idx][time_idx] = demand_to_id[d]

        # conv grid elements to ints
        self.grid_demands = self.grid_demands.astype(int)





    def _calc_flow_completion_times(self):
        flow_completion_times = []
        for flow in self.completed_flow_dicts:
            flow_completion_times.append(flow['time_completed'] - flow['time_arrived'])

        if len(flow_completion_times) == 0:
            average_fct, ninetyninth_percentile_fct = float('inf'), float('inf')
        else:
            average_fct = np.average(np.asarray(flow_completion_times))
            ninetyninth_percentile_fct = np.percentile(np.asarray(flow_completion_times), 99)

        return flow_completion_times, average_fct, ninetyninth_percentile_fct

    def _compute_flow_arrival_metrics(self):
        self.arrived_flow_dicts = self._get_flows_arrived_in_measurement_period() 
        self.flow_times_arrived = [self.arrived_flow_dicts[i]['time_arrived'] for i in range(len(self.arrived_flow_dicts))]
        self.measurement_duration, self.measurement_start_time, self.measurement_end_time = self._get_measurement_times()
        self.num_arrived_flows = len(self.arrived_flow_dicts)

    def _compute_flow_completion_metrics(self):
        self.completed_flow_dicts = self._get_flows_completed_in_measurement_period()
        self.flow_times_completed = [self.completed_flow_dicts[i]['time_completed'] for i in range(len(self.completed_flow_dicts))]
        self.num_completed_flows = len(self.completed_flow_dicts)

        self.fcts, self.average_fct, self.nn_fct = self._calc_flow_completion_times()

    def _compute_flow_dropped_metrics(self):
        self.dropped_flow_dicts = self._get_flows_dropped_in_measurement_period()
        self.num_dropped_flows = len(self.dropped_flow_dicts)
        self.dropped_flow_frac = len(self.dropped_flow_dicts) / len(self.arrived_flow_dicts)

    def _compute_flow_queued_metrics(self):
        self.queued_flow_dicts = self._get_flows_remaining_in_queue_at_end_of_measurement_period()
        self.num_queued_flows = len(self.queued_flow_dicts)

    def _get_measurement_times(self):
        if self.measurement_start_time is None:
            measurement_start_time = min(self.flow_times_arrived)
        elif self.measurement_start_time == 'auto' and self.measurement_end_time == 'auto':
            # both start and end must be assigned simultaneously
            self.measurement_start_time = 0.1 * max(self.flow_times_arrived)
            self.measurement_end_time = 0.9 * max(self.flow_times_arrived)
            # update arrived flows to be within measurement duration
            self.arrived_flow_dicts = self._get_flows_arrived_in_measurement_period() 
            self.flow_times_arrived = [self.arrived_flow_dicts[i]['time_arrived'] for i in range(len(self.arrived_flow_dicts))]
            measurement_start_time = self.measurement_start_time
            measurement_end_time = self.measurement_end_time
        elif self.measurement_start_time == 'auto' and self.measurement_end_time != 'auto':
            self.measurement_start_time = 0.1 * max(self.flow_times_arrived)
            self.arrived_flow_dicts = self._get_flows_arrived_in_measurement_period()
            self.flow_times_arrived = [self.arrived_flow_dicts[i]['time_arrived'] for i in range(len(self.arrived_flow_dicts))]
            measurement_start_time = self.measurement_start_time
        else:
            measurement_start_time = self.measurement_start_time

        if self.measurement_end_time is None:
            measurement_end_time = max(self.flow_times_arrived)
        elif self.measurement_end_time == 'auto' and self.measurement_start_time != 'auto':
            self.measurement_start_time = 0.9 * max(self.flow_times_arrived)
            self.arrived_flow_dicts = self._get_flows_arrived_in_measurement_period()
            self.flow_times_arrived = [self.arrived_flow_dicts[i]['time_arrived'] for i in range(len(self.arrived_flow_dicts))]
            measurement_start_time = self.measurement_start_time
        else:
            measurement_end_time = self.measurement_end_time
        measurement_duration = measurement_end_time - measurement_start_time

        return measurement_duration, measurement_start_time, measurement_end_time

    def _get_flows_remaining_in_queue_at_end_of_measurement_period(self):
        queued_flow_dicts = []

        # create dicts to enable efficient hash searching
        completed_flow_ids = {self.completed_flow_dicts[i]['flow_id']: i for i in range(len(self.completed_flow_dicts))}
        dropped_flow_ids = {self.dropped_flow_dicts[i]['flow_id']: i for i in range(len(self.dropped_flow_dicts))}

        for i in range(len(self.arrived_flow_dicts)):
            flow_id = self.arrived_flow_dicts[i]['flow_id']
            if flow_id not in completed_flow_ids and flow_id not in dropped_flow_ids:
                queued_flow_dicts.append(self.arrived_flow_dicts[i])

        return queued_flow_dicts

    def _get_flows_dropped_in_measurement_period(self):
        '''Find all flows which arrived during measurement period and were dropped.'''
        dropped_flow_dicts = []

        if self.measurement_start_time is None and self.measurement_end_time is None:
            return self.env.dropped_flows

        elif self.measurement_start_time is None and self.measurement_end_time is not None:
            for idx in range(len(self.env.dropped_flows)):
                arr_time = self.env.dropped_flows[idx]['time_arrived']
                if arr_time < self.measurement_end_time:
                    dropped_flow_dicts.append(self.env.dropped_flows[idx])
                else:
                    # cooling down
                    pass

        elif self.measurement_start_time is not None and self.measurement_end_time is None:
            for idx in range(len(self.env.dropped_flows)):
                arr_time = self.env.dropped_flows[idx]['time_arrived']
                if arr_time > self.measurement_start_time:
                    dropped_flow_dicts.append(self.env.dropped_flows[idx])
                else:
                    # warming up
                    pass

        else:
            for idx in range(len(self.env.dropped_flows)):
                arr_time = self.env.dropped_flows[idx]['time_arrived']
                if arr_time < self.measurement_start_time:
                    # warming up
                    pass
                elif arr_time > self.measurement_start_time and arr_time < self.measurement_end_time:
                    # measure
                    dropped_flow_dicts.append(self.env.dropped_flows[idx])
                elif arr_time > self.measurement_start_time and arr_time > self.measurement_end_time:
                    # cooling down
                    pass

        return dropped_flow_dicts 

    def _get_flows_completed_in_measurement_period(self):
        '''Find all flows which arrived during measurement period and were completed.'''
        completed_flow_dicts = []

        if self.measurement_start_time is None and self.measurement_end_time is None:
            return self.env.completed_flows

        elif self.measurement_start_time is None and self.measurement_end_time is not None:
            for idx in range(len(self.env.completed_flows)):
                arr_time = self.env.completed_flows[idx]['time_arrived']
                if arr_time < self.measurement_end_time:
                    completed_flow_dicts.append(self.env.completed_flows[idx])
                else:
                    # cooling down
                    pass

        elif self.measurement_start_time is not None and self.measurement_end_time is None:
            for idx in range(len(self.env.completed_flows)):
                arr_time = self.env.completed_flows[idx]['time_arrived']
                if arr_time > self.measurement_start_time:
                    completed_flow_dicts.append(self.env.completed_flows[idx])
                else:
                    # warming up
                    pass

        else:
            for idx in range(len(self.env.completed_flows)):
                arr_time = self.env.completed_flows[idx]['time_arrived']
                if arr_time < self.measurement_start_time:
                    # warming up
                    pass
                elif arr_time > self.measurement_start_time and arr_time < self.measurement_end_time:
                    # measure
                    completed_flow_dicts.append(self.env.completed_flows[idx])
                elif arr_time > self.measurement_start_time and arr_time > self.measurement_end_time:
                    # cooling down
                    pass

        return completed_flow_dicts 

    def _get_flows_arrived_in_measurement_period(self):
        '''Find flows arrived during measurement period.'''
        flows_arrived = []

        if self.measurement_start_time is None and self.measurement_end_time is None:
            return self.env.arrived_flow_dicts

        elif self.measurement_start_time == 'auto' and self.measurement_end_time == 'auto':
            # assume all arrived for now, will update later
            return self.env.arrived_flow_dicts

        elif self.measurement_start_time is None and self.measurement_end_time is not None:
            for idx in range(len(self.env.arrived_flow_dicts)):
                arr_time = self.env.arrived_flow_dicts[idx]['time_arrived']
                if arr_time < self.measurement_end_time:
                    flows_arrived.append(self.env.arrived_flow_dicts[idx])
                else:
                    # cooling down
                    pass

        elif self.measurement_start_time is not None and self.measurement_end_time is None:
            for idx in range(len(self.env.arrived_flow_dicts)):
                arr_time = self.env.arrived_flow_dicts[idx]['time_arrived']
                if arr_time > self.measurement_start_time:
                    flows_arrived.append(self.env.arrived_flow_dicts[idx])
                else:
                    # warming up
                    pass

        else:
            for idx in range(len(self.env.arrived_flow_dicts)):
                arr_time = self.env.arrived_flow_dicts[idx]['time_arrived']
                if arr_time < self.measurement_start_time:
                    # warming up
                    pass
                elif arr_time > self.measurement_start_time and arr_time < self.measurement_end_time:
                    # measure
                    flows_arrived.append(self.env.arrived_flow_dicts[idx])
                elif arr_time > self.measurement_start_time and arr_time > self.measurement_end_time:
                    # cooling down
                    pass

        return flows_arrived 


    #################################### JOB ##################################
    def _print_job_summary(self):
        print('\n ~* Job Information *~')
        print('Total number of generated jobs passed to env: {}'.format(self.env.demand.num_demands))
        print('Total number of these jobs which arrived during measurement period: {}'.format(len(self.arrived_job_dicts)))
        print('Time first job arrived: {}'.format(min(self.job_times_arrived)))
        print('Time last job arrived: {}'.format(max(self.job_times_arrived)))
        print('Total number of jobs that were completed: {}'.format(len(self.completed_job_dicts)))
        print('Total number of jobs that were dropped: {}'.format(len(self.dropped_job_dicts)))
        print('Total number of jobs that were left in queue at end of measurement period: {}'.format(len(self.queued_job_dicts)))
        print('Average job completion time (JCT): {}'.format(self.average_jct))
        print('99th percentile JCT: {}'.format(self.nn_jct))

    def _compute_job_summary(self):
        self._compute_job_arrival_metrics()
        self._compute_job_completion_metrics()
        self._compute_job_dropped_metrics()
        self._compute_job_queued_metrics()

    def _compute_job_arrival_metrics(self):
        self.arrived_job_dicts = self._get_jobs_arrived_in_measurement_period(self.measurement_start_time, self.measurement_end_time)
        self.job_times_arrived = [self.arrived_job_dicts[i]['time_arrived'] for i in range(len(self.arrived_job_dicts))]
        self.num_arrived_jobs = len(self.arrived_job_dicts)

    def _compute_job_completion_metrics(self):
        self.completed_job_dicts = self._get_jobs_completed_in_measurement_period()
        self.num_completed_jobs = len(self.completed_job_dicts)

        self.jcts, self.average_jct, self.nn_jct = self._calc_job_completion_times()
        
    def _compute_job_dropped_metrics(self):
        self.dropped_job_dicts = self._get_jobs_dropped_in_measurement_period()
        self.num_dropped_jobs = len(self.dropped_job_dicts)

    def _compute_job_queued_metrics(self):
        self.queued_job_dicts = self._get_job_remaining_in_queue_at_end_of_measurement_period()
        self.num_queued_jobs = len(self.queued_job_dicts)

    def _get_jobs_remaining_in_queue_at_end_of_measurement_period(self):
        queued_job_dicts = []

        # create dicts to enable efficient hash searching
        completed_job_ids = {self.completed_job_dicts[i]['job_id']: i for i in range(len(self.completed_job_dicts))}
        dropped_job_ids = {self.dropped_job_dicts[i]['job_id']: i for i in range(len(self.dropped_job_dicts))}

        for i in range(len(self.arrived_job_dicts)):
            job_id = self.arrived_job_dicts[i]['job_id']
            if job_id not in completed_job_ids and job_id not in dropped_job_ids:
                queued_job_dicts.append(self.arrived_job_dicts[i])

        return queued_job_dicts

    def _get_jobs_arrived_in_measurement_period(self):
        '''Find jobs arrived during measurement period.'''
        jobs_arrived = []

        if self.measurement_start_time is None and self.measurement_end_time is None:
            return self.env.arrived_job_dicts

        elif self.measurement_start_time is None and self.measurement_end_time is not None:
            for idx in range(len(self.env.arrived_job_dicts)):
                arr_time = self.env.arrived_job_dicts[idx]['time_arrived']
                if arr_time < self.measurement_end_time:
                    jobs_arrived.append(self.env.arrived_job_dicts[idx])
                else:
                    # cooling down
                    pass

        elif self.measurement_start_time is not None and self.measurement_end_time is None:
            for idx in range(len(self.env.arrived_job_dicts)):
                arr_time = self.env.arrived_job_dicts[idx]['time_arrived']
                if arr_time > self.measurement_start_time:
                    jobs_arrived.append(self.env.arrived_job_dicts[idx])
                else:
                    # warming up
                    pass

        else:
            for idx in range(len(self.env.arrived_job_dicts)):
                arr_time = self.env.arrived_job_dicts[idx]['time_arrived']
                if arr_time < self.measurement_start_time:
                    # warming up
                    pass
                elif arr_time > self.measurement_start_time and arr_time < self.measurement_end_time:
                    # measure
                    jobs_arrived.append(self.env.arrived_job_dicts[idx])
                elif arr_time > self.measurement_start_time and arr_time > self.measurement_end_time:
                    # cooling down
                    pass

        return jobs_arrived 

    def _get_jobs_completed_in_measurement_period(self):
        '''Find all jobs which arrived during measurement period and were completed.'''
        completed_job_dicts = []

        if self.measurement_start_time is None and self.measurement_end_time is None:
            return self.env.completed_job_dicts

        elif self.measurement_start_time is None and self.measurement_end_time is not None:
            for idx in range(len(self.env.completed_jobs)):
                arr_time = self.env.completed_jobs[idx]['time_arrived']
                if arr_time < self.measurement_end_time:
                    completed_job_dicts.append(self.env.completed_jobs[idx])
                else:
                    # cooling down
                    pass

        elif self.measurement_start_time is not None and self.measurement_end_time is None:
            for idx in range(len(self.env.completed_jobs)):
                arr_time = self.env.completed_jobs[idx]['time_arrived']
                if arr_time > self.measurement_start_time:
                    completed_job_dicts.append(self.env.completed_jobs[idx])
                else:
                    # warming up
                    pass

        else:
            for idx in range(len(self.env.completed_jobs)):
                arr_time = self.env.completed_jobs[idx]['time_arrived']
                if arr_time < self.measurement_start_time:
                    # warming up
                    pass
                elif arr_time > self.measurement_start_time and arr_time < self.measurement_end_time:
                    # measure
                    completed_job_dicts.append(self.env.completed_jobs[idx])
                elif arr_time > self.measurement_start_time and arr_time > self.measurement_end_time:
                    # cooling down
                    pass

        return completed_job_dicts 

    def _calc_job_completion_times(self):
        job_completion_times = []
        for job in self.completed_job_dicts:
            job_completion_times.append(job['time_completed'] - job['time_arrived'])

        if len(job_completion_times) == 0:
            average_jct, ninetyninth_percentile_jct = float('inf'), float('inf')
        else:
            average_jct = np.average(np.asarray(job_completion_times))
            ninetyninth_percentile_jct = np.percentile(np.asarray(job_completion_times), 99)

        return job_completion_times, average_jct, ninetyninth_percentile_jct


    def _get_jobs_dropped_in_measurement_period(self):
        '''Find all jobs which arrived during measurement period and were dropped.'''
        dropped_job_dicts = []

        if self.measurement_start_time is None and self.measurement_end_time is None:
            return self.env.dropped_jobs

        elif self.measurement_start_time is None and self.measurement_end_time is not None:
            for idx in range(len(self.env.dropped_jobs)):
                arr_time = self.env.dropped_jobs[idx]['time_arrived']
                if arr_time < self.measurement_end_time:
                    dropped_job_dicts.append(self.env.dropped_jobs[idx])
                else:
                    # cooling down
                    pass

        elif self.measurement_start_time is not None and self.measurement_end_time is None:
            for idx in range(len(self.env.dropped_jobs)):
                arr_time = self.env.dropped_jobs[idx]['time_arrived']
                if arr_time > self.measurement_start_time:
                    dropped_job_dicts.append(self.env.dropped_jobs[idx])
                else:
                    # warming up
                    pass

        else:
            for idx in range(len(self.env.dropped_jobs)):
                arr_time = self.env.dropped_jobs[idx]['time_arrived']
                if arr_time < self.measurement_start_time:
                    # warming up
                    pass
                elif arr_time > self.measurement_start_time and arr_time < self.measurement_end_time:
                    # measure
                    dropped_job_dicts.append(self.env.dropped_jobs[idx])
                elif arr_time > self.measurement_start_time and arr_time > self.measurement_end_time:
                    # cooling down
                    pass

        return dropped_job_dicts 





    









