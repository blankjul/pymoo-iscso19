import os
import sys
import time

import numpy as np

for home in ["/home/vesikary/", "/home/blankjul/workspace/", "/mnt/home/blankjul/workspace/"]:
    sys.path.insert(0, home + "pymoo-iscso19")
    sys.path.insert(0, home + "pymoo")


from iscso19.callback import MyCallback
from iscso19.problem import ISCSO2019
from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize


def solve(seed):
    print(f"Starting seed {seed}")

    folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), "results")

    start = time.time()
    method = get_algorithm("ga",
                           pop_size=100,
                           sampling=get_sampling("int_random"),
                           crossover=get_crossover("int_sbx", prob=1.0, eta=3.0),
                           mutation=get_mutation("int_pm", eta=3.0),
                           eliminate_duplicates=True,
                           callback=MyCallback(folder)
                           )

    res = minimize(ISCSO2019(),
                   method,
                   termination=('n_eval', 200000),
                   seed=seed,
                   verbose=True
                   )
    end = time.time()
    elapsed = end - start

    np.savetxt(os.path.join(folder, f"ga_{seed}.x"), res.pop.get("X").astype(np.int))
    np.savetxt(os.path.join(folder, f"ga_{seed}.f"), res.pop.get("F"))
    np.savetxt(os.path.join(folder, f"ga_{seed}.g"), res.pop.get("G"))

    print(f"Finished seed {seed} - runtime: {elapsed}")


if __name__ == "__main__":

    seed = int(sys.argv[1])
    solve(seed)