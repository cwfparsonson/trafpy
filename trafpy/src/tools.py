from trafpy.src.dists import val_dists 

import numpy as np


def gen_event_times(interarrival_times, 
                    duration_times=None,
                    path_to_save=None):
    if duration_times is None:
        event_times = np.zeros((int(len(interarrival_times))))
    else:
        event_times = np.zeros((int(len(interarrival_times))*2))

    # points in time at which establishments occur
    for i in np.arange(0, int(len(interarrival_times))):
        event_times[i] = event_times[i-1] + interarrival_times[i-1]

    if duration_times is not None:
        # points in time at which take downs occur
        for i in np.arange(int(len(interarrival_times)), 2*int(len(interarrival_times))):
            event_times[i] = event_times[i-int(len(interarrival_times))] + duration_times[i-(int(len(interarrival_times))+1)]
    else:
        # only consider arrival times, dont need take downs
        pass
    
    if path_to_save is not None:
        val_dists.save_data(path_to_save, event_times[:int(len(interarrival_times))])
        if duration_times is not None:
            val_dists.save_data(path_to_save, event_times[int(len(interarrival_times)):])

    return event_times














































