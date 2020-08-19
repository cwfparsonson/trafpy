LOAD_DEMANDS = None
NUM_EPISODES = 1
NUM_K_PATHS = 1
NUM_CHANNELS = 1
NUM_DEMANDS = 10
MIN_FLOW_SIZE = 1 
MAX_FLOW_SIZE = 100 
MIN_NUM_OPS = 50 
MAX_NUM_OPS = 200 
C =  1.5 
MIN_INTERARRIVAL = 1
MAX_INTERARRIVAL = 1e8
SLOT_SIZE = 10000 
MAX_FLOWS = None
MAX_TIME = None 
ENDPOINT_LABEL = 'server'
ENDPOINT_LABELS = [ENDPOINT_LABEL+'_'+str(ep) for ep in range(5)]
PATH_FIGURES = '../figures/'
PATH_PICKLES = '../pickles/demand/tf_graphs/real/'

print('Demand config file imported.')
if ENDPOINT_LABELS is None:
    print('Warning: ENDPOINTS left as None. Will need to provide own networkx \
            graph with correct labelling. To avoid this, specify list of endpoint \
            labels in config.py')

