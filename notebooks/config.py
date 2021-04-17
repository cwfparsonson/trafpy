LOAD_DEMANDS = None
NUM_EPISODES = 1
NUM_K_PATHS = 1
NUM_CHANNELS = 1
NUM_DEMANDS = 2
MIN_FLOW_SIZE = 1 # 1
MAX_FLOW_SIZE = 100 # 100
MIN_NUM_OPS = 50 # 50 10 10
MAX_NUM_OPS = 200 # 200 7000 1000
C =  1.5 # 0.475 1.5
MIN_INTERARRIVAL = 1
MAX_INTERARRIVAL = 1e8
SLOT_SIZE = 1e2 # 0.2
PACKET_SIZE = 1
MAX_FLOWS = 10 # None
MAX_TIME = None # None
ENDPOINT_LABEL = 'server'
ENDPOINT_LABELS = [ENDPOINT_LABEL+'_'+str(ep) for ep in range(8)]
# ENDPOINT_LABELS = None
PATH_FIGURES = '../figures/'
PATH_PICKLES = '../pickles/demand/tf_graphs/real/'

print('Demand config file imported.')
if ENDPOINT_LABELS is None:
    print('Warning: ENDPOINTS left as None. Will need to provide own networkx \
            graph with correct labelling. To avoid this, specify list of endpoint \
            labels in config.py')

