import copy
import os

import numpy as np

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
        np.savetxt(os.path.join(self.folder, f"custom_{algorithm.seed}.status"), np.array([algorithm.n_gen]))

        nth_gen = 200000 / (algorithm.pop_size * self.n_snapshots)
        if algorithm.n_gen % nth_gen == 0:
            store(self.history, algorithm)
            np.save(os.path.join(self.folder, f"custom_{algorithm.seed}"), self.history)
