import numpy as np

from PymoNNto import Behaviour, def_dtype

# should be after or be
from src.configs.plotters import threshold_plotter, activity_plotter


class ActivityBaseHomeostasis(Behaviour):
    def set_variables(self, n):
        self.window_size = self.get_init_attr("window_size", 100, n)
        self.updating_rate = self.get_init_attr("updating_rate", 0.001, n)
        activity_rate = self.get_init_attr("activity_rate", 5, n)
        if isinstance(activity_rate, (float, int)):
            activity_rate = n.get_neuron_vec(mode="ones") * activity_rate / n.size
        """
            It must at an average spikes for at least one time in w/A range
            for penalty we decrease -A/w (-(w/A)^-1) every one step it doesn't spike
        """

        self.activity_step = -activity_rate / self.window_size

        # rename activity_rate for the readability [same ref]
        best_activity = activity_rate
        best_activity += (self.window_size - activity_rate) * self.activity_step

        self.max_activity = self.get_init_attr("max_activity", best_activity, n)
        self.min_activity = self.get_init_attr("min_activity", best_activity, n)

        self.activities = n.get_neuron_vec(mode="zeros")
        self.exhaustion = n.get_neuron_vec(mode="zeros")

    def new_iteration(self, n):
        self.activities += np.where(n.fired, 1, self.activity_step)
        activity_plotter.add(self.activities)
        if (n.iteration % self.window_size) == 0:
            greater = ((self.activities > self.max_activity) * -1).astype(def_dtype)
            smaller = ((self.activities < self.min_activity) * 1).astype(def_dtype)
            greater *= self.activities - self.max_activity
            smaller *= self.min_activity - self.activities

            are_nearly_the_same = np.isclose(self.activities, self.max_activity)
            if are_nearly_the_same.any():
                greater[are_nearly_the_same] = 0
                smaller[are_nearly_the_same] = 0

            change = (
                (greater + smaller)
                * self.updating_rate
                * 0.99 ** (n.iteration // self.window_size)
            )

            self.exhaustion += change
            n.threshold -= self.exhaustion
            threshold_plotter.add(n.threshold)

            # For: Logic for adaptive updating rate (see old trunks)

            self.activities *= 0
            self.exhaustion *= 0
