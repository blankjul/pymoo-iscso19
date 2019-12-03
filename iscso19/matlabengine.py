import os

import matlab.engine


class MatlabEngine:
    __instance = None

    @staticmethod
    def get_instance():
        if MatlabEngine.__instance is None:
            MatlabEngine.__instance = matlab.engine.start_matlab(option="")

        path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'matlab'))
        MatlabEngine.__instance.addpath(path, nargout=0)

        return MatlabEngine.__instance
