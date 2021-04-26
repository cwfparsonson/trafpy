import multiprocessing
from tqdm import tqdm
import time
import random

def do_work(x):
    time.sleep(0.0001*random.randint(1, 1000))
    # time.sleep(x*0.1)

num_repeats = 100
num_tasks_per_repeat = 100
pbar = tqdm(total=num_repeats*num_tasks_per_repeat, miniters=1, desc='Testing', leave=False, smoothing=1e-5)
start = time.time()

for repeat in range(num_repeats):
    pool = multiprocessing.Pool(10, maxtasksperchild=1)
    results = [pool.apply_async(do_work, args=(repeat,), callback=lambda _: pbar.update(1)) for _ in range(num_tasks_per_repeat)]
    pool.close()
    pool.join()
    output = [p.get() for p in results]
    del pool
    jobs = output

end = time.time()
pbar.close()
print('Completedn test in {} s.'.format(end-start))
