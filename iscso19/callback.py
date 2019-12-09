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

    def __init__(self, folder, name="ga") -> None:
        super().__init__()
        self.name = name
        self.history = []
        self.folder = folder

    def notify(self, algorithm):
        best = filter_optimum(algorithm.pop, least_infeasible=True)
        watch = f"| Seed: {algorithm.seed} | Gen: {algorithm.n_gen} | Evals: {algorithm.evaluator.n_eval} | F min: {best.F} | F avg: {algorithm.pop.get('F').mean()} | G1 min: {best.G[0]} | G2 min: {best.G[1]} |"
        with open(os.path.join(self.folder, f"{self.name}_{algorithm.seed}.status"), 'w+') as fp:
            fp.write(watch + "\n")

        if algorithm.n_gen % 100 == 0:
            store(self.history, algorithm)
            np.save(os.path.join(self.folder, f"{self.name}_{algorithm.seed}"), self.history)
