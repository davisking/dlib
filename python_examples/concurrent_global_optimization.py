import logging
from functools import partial
from multiprocessing import Process, Queue
from time import sleep

import dlib
import numpy as np

logger = logging.getLogger('')


class Simulation:
    def __init__(self, alpha, beta, epsilon):
        self.alpha = alpha
        self.epsilon = epsilon

    def simulate(self):
        # Run simulation, return result
        np.random.seed()
        x = np.random.randint(2, 5)
        sleep(x)
        return x


def worker_proc(stratclass, pp, space_params, result_queue, i, space_vals):
    logger = logging.getLogger('')
    logger.error(f"Testing {space_params}: {space_vals}")
    # Add/overwrite problem params with params given from dlib
    for j, key in enumerate(space_params):
        pp[key] = space_vals[j]
    strat = stratclass(**pp)
    result = strat.simulate()
    result_queue.put((i, result))


# Problem parameters to be used in simulation but not optimized over
pp = {'beta': 2}

# Problem parameters to be used in simulation and to be optimizer over
space = {'epsilon': [True, 10, 2000], 'alpha': [False, 0.00001, 0.3]}
params, is_int, lo_bounds, hi_bounds = [], [], [], []
for p, li in space.items():
    params.append(p)
    is_int.append(li[0])
    lo_bounds.append(li[1])
    hi_bounds.append(li[2])

n_sims = 4  # The number of times to sample and test params
n_concurrent = 4  # Number of concurrent procs
assert n_sims >= n_concurrent

spec = dlib.function_spec(bound1=lo_bounds, bound2=hi_bounds, is_integer=is_int)
optimizer = dlib.global_function_search(spec)
result_queue = Queue()
simproc = partial(worker_proc, Simulation, pp, params, result_queue)
results = np.zeros((n_sims, len(params) + 1))
evals = [None] * n_sims

logger.error(f"Dlib hopt for {n_sims} sims with {n_concurrent} procs for"
             f" on params {params} with bounds {lo_bounds}, {hi_bounds}")
# Spawn initial processes
for i in range(n_concurrent):
    eeval = optimizer.get_next_x()
    next_x = list(eeval.x)
    evals[i] = eeval
    results[i][1:] = next_x
    Process(target=simproc, args=(i, next_x)).start()

for j in range(n_concurrent, n_sims):
    # Block until a result is ready
    i, result = result_queue.get()
    evals[i].set(result)
    results[i][0] = result

    # Spawn a new process
    eeval = optimizer.get_next_x()
    next_x = list(eeval.x)
    evals[j] = eeval
    results[j][1:] = next_x
    Process(target=simproc, args=(j, next_x)).start()

# Get remaining results
for j in range(n_concurrent):
    i, result = result_queue.get()
    evals[i].set(result)
    results[i][0] = result

logger.error(f"[Result, {params}]\n{results}")
