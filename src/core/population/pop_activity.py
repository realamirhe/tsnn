import numpy as np

from PymoNNto import Behaviour


class PopulationBaseActivity(Behaviour):
    def set_variables(self, n):
        window_size = self.get_init_attr("window_size", 10, n)
        n.A_history = np.zeros(window_size)
        n.A = 0.0

    def new_iteration(self, n):
        n.A_history[1:] = n.A_history[0:-1]
        n.A_history[0] = np.sum(n.fired) / n.size  # An
        n.A = np.average(n.A_history)
