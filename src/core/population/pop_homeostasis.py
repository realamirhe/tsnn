import numpy as np

from PymoNNto import Behaviour


class PopulationBaseHomeostasis(Behaviour):
    def set_variables(self, n):
        self.window_size = self.get_init_attr("window_size", 100, n)
        self.updating_rate = self.get_init_attr("updating_rate", 0.001, n)
        population_count = self.get_init_attr("population_count", 1, n)
        self.activity_rate = (
            self.get_init_attr("activity_rate", 5, n) / population_count
        )

    def new_iteration(self, n):
        if (n.iteration % self.window_size) == 0:
            change = (n.A - self.activity_rate) * self.updating_rate
            n.threshold += change * np.power(0.99, n.iteration // self.window_size)
