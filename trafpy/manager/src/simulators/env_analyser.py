class EnvAnalyser:

    def __init__(self, env):
        '''
        envs (obj): Environment/simulation object to analyse.
        '''
        self.env = env

    def compute_flow_summary(self, warm_up_end_time=None, cool_down_start_time=None, print_summary=False):
        '''
        warm_up_end_time (int, float): Simulation time at which to begin recording
            metrics etc.
        cool_down_start_time (int, float): Simulation time at which to stop recording
            metrics etc.
        '''
        self.arrived_flow_dicts = self._get_flows_arrived_in_measurement_period(warm_up_end_time, cool_down_start_time) 
        self.flow_times_arrived = [self.arrived_flow_dicts[i]['time_arrived'] for i in range(len(self.arrived_flow_dicts))]
        self.measurement_duration, self.measurement_start_time, self.measurement_end_time = self._get_measurement_times(warm_up_end_time, cool_down_start_time)
        self.num_arrived_flows = len(self.arrived_flow_dicts)

        self.completed_flow_dicts = self._get_flows_completed_in_measurement_period(warm_up_end_time, cool_down_start_time)
        self.flow_times_completed = [self.completed_flow_dicts[i]['time_completed'] for i in range(len(self.completed_flow_dicts))]
        self.num_completed_flows = len(self.completed_flow_dicts)

    def _get_flows_completed_in_measurement_period(self, warm_up_end_time, cool_down_start_time):
        '''Find all flows which arrived during measurement period and were completed.'''
        completed_flow_dicts = []

        if warm_up_end_time is None and cool_down_start_time is None:
            return self.env.completed_flows

        elif warm_up_end_time is None and cool_down_start_time is not None:
            for idx in range(len(self.env.completed_flows)):
                arr_time = self.env.completed_flows[idx]['time_arrived']
                if arr_time < cool_down_start_time:
                    completed_flow_dicts.append(env.completed_flows[idx])
                else:
                    # cooling down
                    pass

        elif warm_up_end_time is not None and cool_down_start_time is None:
            for idx in range(len(self.env.completed_flows)):
                arr_time = self.env.completed_flows[idx]['time_arrived']
                if arr_time > warm_up_end_time:
                    completed_flow_dicts.append(env.completed_flows[idx])
                else:
                    # warming up
                    pass

        else:
            for idx in range(len(self.env.completed_flows)):
                arr_time = self.env.completed_flows[idx]['time_arrived']
                if arr_time < warm_up_end_time:
                    # warming up
                    pass
                elif arr_time > warm_up_end_time and arr_time < cool_down_start_time:
                    # measure
                    completed_flow_dicts.append(env.completed_flows[idx])
                elif arr_time > warm_up_end_time and arr_time > cool_down_start_time:
                    # cooling down
                    pass

        return completed_flow_dicts 



    def _get_measurement_times(self, warm_up_end_time, cool_down_start_time):
        if warm_up_end_time is None:
            measurement_start_time = min(self.flow_times_arrived)
        if cool_down_start_time is None:
            measurement_end_time = max(self.flow_times_arrived)
        measurement_duration = measurement_end_time - measurement_start_time

        return measurement_duration, measurement_start_time, measurement_end_time


    def _get_flows_arrived_in_measurement_period(self, warm_up_end_time, cool_down_start_time):
        '''Find flows arrived during measurement period.'''
        flows_arrived = []

        if warm_up_end_time is None and cool_down_start_time is None:
            return self.env.arrived_flow_dicts

        elif warm_up_end_time is None and cool_down_start_time is not None:
            for idx in range(len(self.env.arrived_flow_dicts)):
                arr_time = self.env.arrived_flow_dicts[idx]['time_arrived']
                if arr_time < cool_down_start_time:
                    flows_arrived.append(env.arrived_flow_dicts[idx])
                else:
                    # cooling down
                    pass

        elif warm_up_end_time is not None and cool_down_start_time is None:
            for idx in range(len(self.env.arrived_flow_dicts)):
                arr_time = self.env.arrived_flow_dicts[idx]['time_arrived']
                if arr_time > warm_up_end_time:
                    flows_arrived.append(env.arrived_flow_dicts[idx])
                else:
                    # warming up
                    pass

        else:
            for idx in range(len(self.env.arrived_flow_dicts)):
                arr_time = self.env.arrived_flow_dicts[idx]['time_arrived']
                if arr_time < warm_up_end_time:
                    # warming up
                    pass
                elif arr_time > warm_up_end_time and arr_time < cool_down_start_time:
                    # measure
                    flows_arrived.append(env.arrived_flow_dicts[idx])
                elif arr_time > warm_up_end_time and arr_time > cool_down_start_time:
                    # cooling down
                    pass

        return flows_arrived 


        return flows_completed 



    



    def cal_total_info_arrived(self, env):
        flow_sizes_arrived = []
        num_arrived_flows = len(env.arrived_flows.keys())

        for idx in range(num_arrived_flows):
            arr_time = env.arrived_flow_dicts[idx]['time_arrived']
            if arr_time < self.warm_up_end_time:
                # not finished warming up sim env
                pass
            elif arr_time > self.warm_up_end_time and arr_time < self.cool_down_start_time:
                # measure
                flow_sizes_arrived.append(env.arrived_flow_dicts[idx]['size'])
            elif arr_time > self.warm_up_end_time and arr_time > self.cool_down_start_time:
                # cooling down
                pass

        total_info_arrived = sum(flow_sizes_arrived)

        return flow_sizes_arrived, total_info_arrived
    
    def calc_network_load_abs(self, env):
        pass






