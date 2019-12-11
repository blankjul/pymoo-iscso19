import numpy as np
import matplotlib.pyplot as plt
import sys, os, time


root = os.path.dirname(os.getcwd())
sys.path.insert(0, root)

# Pymoo imports
from pymoo.algorithms.nsga2 import RankAndCrowdingSurvival
from pymoo.model.survival import Survival, split_by_feasibility
from pymoo.algorithms.so_genetic_algorithm import GA, FitnessSurvival
from pymoo.factory import get_crossover, get_mutation, get_sampling
from pymoo.model.algorithm import filter_optimum
from pymoo.model.population import Population
from pymoo.model.sampling import Sampling
from pymoo.model.termination import Termination
from pymoo.optimize import minimize

# ISCSO imports 
from iscso19.callback import MyCallback
from iscso19.display import MyDisplay
from iscso19.problem import ISCSO2019
from iscso19.methods.restarts_nds import MyGeneticAlgorithm, RestartDisplay


seed = 310

print(f"Starting seed {seed}")

start = time.time()
method = MyGeneticAlgorithm(
    pop_size=200,
    n_offsprings=100,
    sampling=get_sampling("int_random"),
    crossover=get_crossover("int_sbx", prob=1.0, eta=3.0),
    mutation=get_mutation("int_pm", eta=3.0),
    eliminate_duplicates=True,
    display=RestartDisplay()
)

res = minimize(ISCSO2019(),
               method,
               termination=('n_eval', 200000),
               seed=seed,
               verbose=True, 
               save_history=True
               )
end = time.time()
elapsed = end - start

print(f"Finished seed {seed} - runtime: {elapsed}")

print("Results")
print(f"Final Function Value: {res.F[0]} - Constraint Values: {', '.join(map(str, res.G))}")