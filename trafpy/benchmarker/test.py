import multiprocessing
from tqdm import tqdm
import time
import random
import functools
# multiprocessing.set_start_method('fork')
# from multiprocessing import set_start_method
# set_start_method("spawn")


def do_work(x):
    time.sleep(0.00001*random.randint(1, 1000))


def update_pbar(_pbar):
    _pbar.update(1)

if __name__ == '__main__':
    num_repeats = 100 # 100
    num_tasks_per_repeat = 100 # 100
    pbar = tqdm(total=num_repeats*num_tasks_per_repeat, 
                miniters=1, 
                # mininterval=1,
                maxinterval=1,
                desc='Testing', 
                leave=False, 
                smoothing=0)
    start = time.time()

    for repeat in range(num_repeats):
        pool = multiprocessing.Pool(10, maxtasksperchild=1)
        results = [pool.apply_async(do_work, args=(repeat,), callback=lambda _: pbar.update(1)) for _ in range(num_tasks_per_repeat)]
        # results = [pool.apply_async(do_work, args=(repeat,), callback=functools.partial(pbar.update, 1)) for _ in range(num_tasks_per_repeat)]
        # results = [pool.apply_async(do_work, args=(repeat,), callback=functools.partial(update_pbar, pbar)) for _ in range(num_tasks_per_repeat)]
        pool.close()
        pool.join()
        output = [p.get() for p in results]
        del pool
        jobs = output

    end = time.time()
    pbar.close()
    print('Completed test in {} s.'.format(end-start))
