import os
import sys

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

for home in ["/home/vesikary/", "/home/blankjul/", "/mnt/home/blankjul/workspace/"]:
    sys.path.append(home + "pymoo-iscso19")

from iscso19.callback import MyCallback
from pymoo.algorithms.so_genetic_algorithm import GA

import numpy as np
from pymoo.factory import get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize

from iscso19.problem import ISCSO2019
import time


class MyGeneticAlgorithm(GA):

    def _initialize(self):

        def sample(n):
            pop = get_sampling("int_random").do(self.problem, n)
            self.evaluator.eval(self.problem, pop, algorithm=self)
            return pop


        self.pop = sample(100)
        self._each_iteration(self)

        for k in range(19):
            _pop = sample(100)

            pop = self.pop.merge(_pop)
            I = NonDominatedSorting().do(pop.get("G"), only_non_dominated_front=True)
            self.pop = pop[I]
            self._each_iteration(self)




        print("test")


def solve(seed):
    print(f"Starting seed {seed}")

    folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), "results")

    start = time.time()
    method = MyGeneticAlgorithm(
        pop_size=20,
        sampling=None,
        crossover=get_crossover("int_sbx", prob=1.0, eta=3.0),
        mutation=get_mutation("int_pm", eta=3.0),
        eliminate_duplicates=True,
        callback=MyCallback(folder)
    )

    res = minimize(ISCSO2019(),
                   method,
                   termination=('n_eval', 20),
                   seed=seed,
                   verbose=True
                   )
    end = time.time()
    elapsed = end - start

    np.savetxt(os.path.join(folder, f"custom_{seed}.x"), res.pop.get("X").astype(np.int))
    np.savetxt(os.path.join(folder, f"custom_{seed}.f"), res.pop.get("F"))
    np.savetxt(os.path.join(folder, f"custom_{seed}.g"), res.pop.get("G"))

    print(f"Finished seed {seed} - runtime: {elapsed}")


if __name__ == "__main__":
    seed = int(sys.argv[1])
    solve(seed)
