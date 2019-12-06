import numpy as np

from pymoo.util.display import SingleObjectiveDisplay


class MyDisplay(SingleObjectiveDisplay):

    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        self.output.append("F", np.min(algorithm.pop.get("_F")))
        self.output.append("G1", np.min(algorithm.pop.get("_G")[:, 0]))
        self.output.append("G2", np.min(algorithm.pop.get("_G")[:, 1]))
