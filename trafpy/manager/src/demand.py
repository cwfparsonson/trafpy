from trafpy.generator.src import builder 
import sys

class Demand:
    def __init__(self,
                 demand_data):

        self.reset(demand_data)

    def reset(self, demand_data):
        self.demand_data = demand_data
        self.num_demands = self.get_num_demands(self.demand_data)

        if 'job_id' in demand_data:
            self.job_centric = True
        else:
            self.job_centric = False

        self.num_control_deps, self.num_data_deps, self.num_flows = self.get_num_deps(demand_data)

    def get_slots_dict(self, slot_size):
        return builder.construct_demand_slots_dict(demand_data=self.demand_data,
                                                   slot_size=slot_size)

    def get_num_demands(self, demand_data):
        if 0 in demand_data['establish']:
            takedowns_present = True
        else:
            takedowns_present = False

        if takedowns_present:
            # half events are takedowns for demand establishments
            num_demands = int(len(demand_data['establish'])/2)
        else:
            # all events are new demands
            num_demands = int(len(demand_data['establish']))

        return num_demands

    def get_num_deps(self, demand_data):
        num_control_deps,num_data_deps,num_flows = 0, 0, 0

        if self.job_centric:
            # calc deps
            for job in demand_data['job']:
                for op in job.nodes:
                    flows = job.out_edges(op)
                    for flow in flows:
                        flow_stats = job.get_edge_data(flow[0],flow[1])
                        src = job.nodes[flow[0]]['attr_dict']['machine']
                        dst = job.nodes[flow[1]]['attr_dict']['machine']
                        if flow_stats['attr_dict']['dependency_type'] == 'data_dep':
                            num_data_deps+=1
                            if src != dst:
                                num_flows+=1
                        else:
                            num_control_deps+=1

        else:
            # 1 demand == 1 flow, therefore no dependencies & each demand == flow
            num_flows = self.num_demands
        
        return num_control_deps, num_data_deps, num_flows

