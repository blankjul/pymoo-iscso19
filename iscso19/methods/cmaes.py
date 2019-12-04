import copy
import os
import sys

from pymoo.algorithms.so_cmaes import CMAES, CMAESDisplay
from pymoo.util.normalization import normalize

for home in ["/home/vesikary/", "/home/blankjul/", "/mnt/home/blankjul/workspace/"]:
    sys.path.append(home + "pymoo-iscso19")

import numpy as np
from pymoo.model.callback import Callback
from pymoo.optimize import minimize

from iscso19.problem import ISCSO2019
import time


class MyDisplay(CMAESDisplay):

    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        self.output.append("F", np.min(algorithm.pop.get("_F")))
        self.output.append("G1", np.min(algorithm.pop.get("_G")[:, 0]))
        self.output.append("G2", np.min(algorithm.pop.get("_G")[:, 1]))


class PenaltyProblem(ISCSO2019):

    def __init__(self, penalty):
        super().__init__()
        self.n_constr = 0
        self.penalty = penalty

    def _evaluate(self, X, out, *args, algorithm=None, **kwargs):
        super()._evaluate(X, out, *args, **kwargs)

        out["_F"] = out["F"]
        out["_G"] = out["G"]

        F = normalize(out["F"], 20000, 60000)
        G = normalize(out["G"], np.array([0.0, 0.0]), np.array([2000.0, 200.0]))


        del out["G"]
        out["F"] = F + (self.penalty * G).sum(axis=1)[:, None]


def store(history, algorithm):
    hist, _callback = algorithm.history, algorithm.callback
    algorithm.history, algorithm.callback = None, None

    obj = copy.deepcopy(algorithm)
    algorithm.history = hist
    algorithm.callback = _callback

    history.append(obj)


class MyCallback(Callback):

    def __init__(self, folder, n_snapshots=1000) -> None:
        super().__init__()
        self.n_snapshots = n_snapshots
        self.history = []
        self.folder = folder

    def notify(self, algorithm):
        np.savetxt(os.path.join(self.folder, f"cmaes_{algorithm.seed}.status"), np.array([algorithm.n_gen]))

        nth_gen = 200000 / (algorithm.pop_size * self.n_snapshots)
        if algorithm.n_gen % nth_gen == 0:
            store(self.history, algorithm)
            np.save(os.path.join(self.folder, f"cmaes_{algorithm.seed}"), self.history)


def solve(seed):
    print(f"Starting seed {seed}")

    folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), "results")

    start = time.time()

    problem = PenaltyProblem(0.1)

    method = CMAES(
        integer_variables=list(range(problem.n_var)),
        display=MyDisplay(),
        parallelize=True)

    res = minimize(problem,
                   method,
                   termination=('n_eval', 200000),
                   seed=seed,
                   verbose=True
                   )
    end = time.time()
    elapsed = end - start

    np.savetxt(os.path.join(folder, f"cmaes_{seed}.x"), res.pop.get("X").astype(np.int))
    np.savetxt(os.path.join(folder, f"cmaes_{seed}.f"), res.pop.get("_F"))
    np.savetxt(os.path.join(folder, f"cmaes_{seed}.g"), res.pop.get("_G"))

    print(f"Finished seed {seed} - runtime: {elapsed}")


if __name__ == "__main__":
    # " ".join([str(e) for e in pop.get("X")[1].tolist()])

    seed = int(sys.argv[1])
    solve(seed)

    # seeds = np.arange(60)
    # with Pool(4) as p:
    #     p.map(solve, seeds)
