import numpy as np

from PymoNNto import Behaviour


class PopulationTraceHistory(Behaviour):
    def set_variables(self, n):
        window_size = self.get_init_attr("window_size", None, n)
        n.alpha = np.zeros(window_size)

    def new_iteration(self, n):
        n.alpha[1:] = n.alpha[0:-1]
