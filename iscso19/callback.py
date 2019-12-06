import copy
import os

import numpy as np
from pymoo.model.algorithm import filter_optimum

from pymoo.model.callback import Callback


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
        best = filter_optimum(algorithm.pop, least_infeasible=True)
        watch = f"| Seed: {algorithm.seed} | Gen: {algorithm.n_gen} | Evals: {algorithm.evaluator.n_eval} | F min: {best.F} | F avg: {algorithm.pop.get('F').mean()} | G1 min: {best.G[0]} | G2 min: {best.G[1]} |"
        with open(os.path.join(self.folder, f"custom_{algorithm.seed}.status"), 'w+') as fp:
            fp.write(watch)

        # np.savetxt(, np.array([algorithm.n_gen]))

        if algorithm.n_gen % 200 == 0:
            store(self.history, algorithm)
            np.save(os.path.join(self.folder, f"custom_{algorithm.seed}"), self.history)
