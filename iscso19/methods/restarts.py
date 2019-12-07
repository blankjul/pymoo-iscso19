import copy
import os
import sys
import time

import numpy as np

for home in ["/home/vesikary/", "/home/blankjul/workspace/", "/mnt/home/blankjul/workspace/"]:
    sys.path.insert(0, home + "pymoo-iscso19")
    sys.path.insert(0, home + "pymoo")


from iscso19.callback import MyCallback
from iscso19.display import MyDisplay
from iscso19.problem import ISCSO2019
from pymoo.algorithms.so_genetic_algorithm import GA, FitnessSurvival
from pymoo.factory import get_crossover, get_mutation, get_sampling
from pymoo.model.algorithm import filter_optimum
from pymoo.model.population import Population
from pymoo.model.sampling import Sampling
from pymoo.model.termination import Termination
from pymoo.optimize import minimize


class RestartDisplay(MyDisplay):

    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        self.output.append("no_impr", algorithm.no_improvement)
        self.output.append("pop_size", algorithm.pop_size)


class UntilFeasibleFoundTermination(Termination):

    def __init__(self, n_max_evals) -> None:
        super().__init__()
        self.n_max_evals = n_max_evals

    def _do_continue(self, algorithm):
        return not (np.any(algorithm.pop.get("feasible")) or algorithm.evaluator.n_eval >= self.n_max_evals)



class BiasedSampling(Sampling):

    def __init__(self, X, perc=0.1) -> None:
        super().__init__()
        self.X = X
        self.perc = perc

    def do(self, problem, n_samples, **kwargs):
        cpy = copy.deepcopy(problem)
        _range = (problem.xu - problem.xl)
        cpy.xl = np.maximum(np.ceil(self.X - _range * self.perc / 2).astype(np.int), problem.xl)
        cpy.xu = np.minimum(np.ceil(self.X + _range * self.perc / 2).astype(np.int), problem.xu)
        return get_sampling("int_random").do(cpy, n_samples - 1).merge(Population().new("X", np.atleast_2d(self.X)))


class MyGeneticAlgorithm(GA):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.no_improvement = 0

    def _initialize_disabled(self):

        # first just explore with 1000 solutions to find some that are not too infeasible
        pop = get_sampling("int_random").do(self.problem, 1000)
        self.evaluator.eval(self.problem, pop, algorithm=self)
        self.pop = FitnessSurvival().do(self.problem, pop, 5)
        self._each_iteration(self)

        best = None

        # for each of the top solutions find now some that are indeed feasible
        for ind in self.pop:

            ga = GA(pop_size=100,
                    sampling=BiasedSampling(ind.X, 0.2),
                    crossover=get_crossover("int_sbx", prob=1.0, eta=3.0),
                    mutation=get_mutation("int_pm", eta=3.0),
                    display=MyDisplay()
                    )

            ret = minimize(self.problem,
                           ga,
                           UntilFeasibleFoundTermination(10000),
                           verbose=True
                           )

            self.evaluator.n_eval += ret.algorithm.evaluator.n_eval

            if best is None:
                best = ret.pop
            else:
                best = best.merge(ret.pop)

        self.pop = best

    def next(self):
        if self.no_improvement > 100:
            # self.pop_size = min(200, int(self.pop_size + self.pop_size // 2))
            
            self.pop = BiasedSampling(self.opt.X, 0.2).do(self.problem, self.pop_size)
            self.evaluator.eval(self.problem, self.pop, algorithm=self)
            self.no_improvement = -1
        else:
            self._next()

        _opt = filter_optimum(self.pop, least_infeasible=True)

        if self.opt is not None and np.all(_opt.F == self.opt.F):
            self.no_improvement += 1
        else:
            self.no_improvement = 0

        self.opt = _opt

        self.n_gen += 1
        self._each_iteration(self)


def solve(seed):
    print(f"Starting seed {seed}")

    folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), "results")

    start = time.time()
    method = MyGeneticAlgorithm(
        pop_size=100,
        sampling=get_sampling("int_random"),
        crossover=get_crossover("int_sbx", prob=1.0, eta=3.0),
        mutation=get_mutation("int_pm", eta=3.0),
        eliminate_duplicates=True,
        callback=MyCallback(folder, name="restarts_only"),
        display=RestartDisplay()
    )

    res = minimize(ISCSO2019(),
                   method,
                   termination=('n_eval', 200000),
                   seed=seed,
                   verbose=True
                   )
    end = time.time()
    elapsed = end - start

    np.savetxt(os.path.join(folder, f"restarts_only_{seed}.x"), res.pop.get("X").astype(np.int))
    np.savetxt(os.path.join(folder, f"restarts_only_{seed}.f"), res.pop.get("F"))
    np.savetxt(os.path.join(folder, f"restarts_only_{seed}.g"), res.pop.get("G"))

    print(f"Finished seed {seed} - runtime: {elapsed}")


if __name__ == "__main__":
    seed = int(sys.argv[1])
    solve(seed)
