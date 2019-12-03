import sys

sys.path.append("/home/vesikary/pymoo-iscso19")

import numpy as np
from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling
from pymoo.model.callback import Callback
from pymoo.optimize import minimize

from iscso19.problem import ISCSO2019
from multiprocessing import Pool
import time

class MyCallback(Callback):

    def __init__(self) -> None:
        super().__init__()

    def notify(self, algorithm):
        if algorithm.n_gen % 10 == 0:
            print("Saving")
            np.savetxt(f"results/ga_{algorithm.seed}.x", np.array(algorithm.pop.get("X")))


def solve(seed):
    print(f"Starting seed {seed}")
    start = time.time()
    method = get_algorithm("ga",
                           pop_size=40,
                           sampling=get_sampling("int_random"),
                           crossover=get_crossover("int_sbx", prob=1.0, eta=3.0),
                           mutation=get_mutation("int_pm", eta=3.0),
                           eliminate_duplicates=True,
                           callback=MyCallback()
                           )

    res = minimize(ISCSO2019(),
                   method,
                   termination=('n_eval', 200000),
                   seed=seed,
                   verbose=True
                   )
    end = time.time()
    elapsed = end-start
    np.savetxt(f"results/ga_{seed}.x", np.array(res.pop.get("X")))
    np.savetxt(f"results/ga_{seed}.f", np.array(res.pop.get("F")))
    print(f"Finished seed {seed} - runtime: {elapsed}")

if __name__ == "__main__":
    seeds = np.arange(60)
    with Pool(4) as p: 
        p.map(solve, seeds)
