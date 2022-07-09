import numpy as np

from PymoNNto import Behaviour


class PopulationBaseActivity(Behaviour):
    def set_variables(self, n):
        window_size = self.get_init_attr("window_size", 10, n)
        self.activity_count_history = np.zeros(window_size)
        n.A = 0.0

    def new_iteration(self, n):
        self.activity_count_history[1:] = self.activity_count_history[0:-1]
        self.activity_count_history[0] = np.sum(n.fired)
        n.A = np.average(self.activity_count_history)
