import numpy as np

from PymoNNto import Behaviour


class PopulationBaseActivity(Behaviour):
    def set_variables(self, n):
        n.A = 0.0

    def new_iteration(self, n):
        n.A += np.sum(n.fired) / n.size  # An
