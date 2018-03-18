import logging
from functools import partial
from multiprocessing import Pool

import dlib

logger = logging.getLogger('')


class Simulation:
    def __init__(self, alpha, beta, epsilon):
        pass

    def simulate(self):
        return 1


def dlib_proc(stratclass, pp, space_params, space_vals):
    logger = logging.getLogger('')
    logger.error(f"Testing {space_params}: {space_vals}")
    for j, key in enumerate(space_params):
        pp[key] = space_vals[j]
    strat = stratclass(**pp)
    result = strat.simulate()
    return result


pp = {'beta': 2}

space = {'epsilon': [True, 10, 2000], 'alpha': [False, 0.00001, 0.3]}
params, is_int, lo_bounds, hi_bounds = [], [], [], []
for p, li in space.items():
    params.append(p)
    is_int.append(li[0])
    lo_bounds.append(li[1])
    hi_bounds.append(li[2])

spec = dlib.function_spec(bound1=lo_bounds, bound2=hi_bounds, is_integer=is_int)
n = 3  # The number of times to sample and test params
n_concurrent = 2  # Number of concurrent procs
optimizer = dlib.global_function_search(spec)
results = []
logger.error(f"Dlib hopt for {n} iterations with {n_concurrent} procs for"
             f" a total of {n*n_concurrent} simulations,"
             f" on params {params} with bounds {lo_bounds}, {hi_bounds}")
simproc = partial(dlib_proc, Simulation, pp, params)
for i in range(n):
    evals = []
    space_vals = []
    for _ in range(n_concurrent):
        n = optimizer.get_next_x()
        evals.append(n)
        space_vals.append(list(n.x))
    # self.logger.error(f"\nIter {i}, testing {params}: {space_vals}")
    with Pool(n_concurrent) as p:
        temp_results = p.map(simproc, space_vals)
    for i, s_eval in enumerate(evals):
        s_eval.set(temp_results[i])
    results.extend([(r, v) for r, v in zip(temp_results, space_vals)])

results.sort()
logger.error(f"[Result, {params}]\n{results}")
"""
Dlib hopt for 3 iterations with 2 procs for a total of 6 simulations, on params ['epsilon', 'alpha'] with bounds [10, 1e-05], [2000, 0.3]
Testing ['epsilon', 'alpha']: [1005.0, 0.150005]
Testing ['epsilon', 'alpha']: [474.0, 0.2287974675814215]
Testing ['epsilon', 'alpha']: [301.0, 0.08454774895245444]
Testing ['epsilon', 'alpha']: [892.0, 0.04051983151878068]
Testing ['epsilon', 'alpha']: [1005.0, 0.181255]
Testing ['epsilon', 'alpha']: [1047.0, 0.22477007832395646]
[Result, ['epsilon', 'alpha']]
[(1, [301.0, 0.08454774895245444]), (1, [474.0, 0.2287974675814215]), (1, [892.0, 0.04051983151878068]), (1, [1005.0, 0.150005]), (1, [1005.0, 0.181255]), (1, [1047.0, 0.22477007832395646])]
"""
