from iscso19.problem import ISCSO2019
from pymoo.algorithms.so_cmaes import CMAES
from pymoo.optimize import minimize

problem = ISCSO2019()

algorithm = CMAES(
    integer_variables=list(range(problem.n_var)))


res = minimize(problem,
               algorithm,
               seed=1,
               verbose=False)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))