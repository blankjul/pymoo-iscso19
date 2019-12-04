import matlab
import numpy as np
from pymoo.model.problem import Problem

from iscso19.matlabengine import MatlabEngine


class ISCSO2019(Problem):

    def __init__(self):
        self.n_sections = 260
        self.n_coords = 10

        super().__init__(n_var=self.n_sections + self.n_coords, n_obj=1, type_var=np.int)

        self.n_constr = 2
        self.xl = np.concatenate([np.ones(self.n_sections), np.full(self.n_coords, -25000)]).astype(np.int)
        self.xu = np.concatenate([np.full(self.n_sections, 37), np.full(self.n_coords, 3500)]).astype(np.int)

    def _evaluate(self, X, out, *args, **kwargs):
        eng = MatlabEngine.get_instance()

        mat_X = matlab.int32(X.astype(np.int).tolist())
        f, g1, g2 = eng.evaluate(mat_X, nargout=3)

        out["F"] = np.atleast_2d(np.array(f))
        out["G"] = np.column_stack([g1, g2])


if __name__ == '__main__':

    X = np.ones((10, 270))

    problem = ISCSO2019()

    f, cv, g = problem.evaluate(X, return_values_of=["F", "CV", "G"])

    print(f, cv, g)