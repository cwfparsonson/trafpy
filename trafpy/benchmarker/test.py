import multiprocessing
from tqdm import tqdm
import time

def do_work(x):
    time.sleep(x*0.1)

num_repeats = 100
num_tasks_per_repeat = 100
pbar = tqdm(total=num_repeats*num_tasks_per_repeat, miniters=1)
for repeat in range(num_repeats):
    pool = multiprocessing.Pool(10, maxtasksperchild=1)
    results = [pool.apply_async(do_work, args=(repeat,), callback=lambda _: pbar.update(1)) for _ in range(num_tasks_per_repeat)]
    pool.close()
    pool.join()
    del pool
pbar.close()
