import numpy as np

from iscso19.problem import ISCSO2019
from pymoo.algorithms.so_cmaes import CMAES, CMAESDisplay
from pymoo.optimize import minimize

problem = ISCSO2019(penalty=1e16)


class MyDisplay(CMAESDisplay):

    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        self.output.append("F", np.min(algorithm.pop.get("_F")))
        self.output.append("G1", np.min(algorithm.pop.get("_G")[:, 0]))
        self.output.append("G2", np.min(algorithm.pop.get("_G")[:, 1]))


algorithm = CMAES(
    integer_variables=list(range(problem.n_var)),
    parallelize=True,
    display=MyDisplay())

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=True)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
