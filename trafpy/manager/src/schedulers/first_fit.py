from trafpy.manager.src.schedulers.schedulertoolbox import SchedulerToolbox




class FirstFit:

    def __init__(self, 
                 Graph, 
                 RWA, 
                 slot_size, 
                 packet_size=300, 
                 time_multiplexing=True, 
                 debug_mode=False, 
                 scheduler_name='first_fit'):
        self.scheduler = SchedulerToolbox(Graph, RWA, slot_size, time_multiplexing, debug_mode)
        self.scheduler_name = scheduler_name

    def get_action(self, observation, print_processing_time=False):
        chosen_flows = self.get_scheduler_action(observation)
        action = {'chosen_flows': chosen_flows}

        return action

    def get_scheduler_action(self, observation):
        # update scheduler network & new flow states
        self.scheduler.update_network_state(observation, hide_child_dependency_flows=True)

        # choose which flow(s) to schedule for this time slot
        chosen_flows = []
        for ep in self.scheduler.SchedulerNetwork.graph['endpoints']:
            queues = self.scheduler.SchedulerNetwork.nodes[ep]
            for queue in queues.keys():
                if len(queues[queue]['queued_flows']) == 0:
                    # no flows queued, move to next queue
                    continue
                else:
                    # queued flow(s) present
                    for flow in queues[queue]['queued_flows']:
                        # fit as much of flow into first channel and path available
                        if self.scheduler.debug_mode:
                            print('\nAttempting to establish flow {}'.format(flow))
                        flow = self.scheduler.init_paths_and_packets(flow)
                        establish_flow, p, c, packets_this_slot = self.scheduler.look_for_available_lightpath(flow, chosen_flows)
                        if establish_flow:
                            # path and channel available, schedule flow
                            flow['path'], flow['channel'], flow['packets_this_slot'] = p, c, packets_this_slot
                            chosen_flows.append(flow)
                            self.scheduler.set_up_connection(flow)
                            if self.scheduler.debug_mode:
                                print('Flow can be established with params {}'.format(flow))
                        else:
                            # no flow or channel available. Doing first fit so no contention resolution -> do not schedule this flow
                            if self.scheduler.debug_mode:
                                print('Flow could not be established.')
                            pass

        return chosen_flows

























