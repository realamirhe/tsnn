import numpy as np

from PymoNNto import Behaviour, def_dtype


# should be after or be
class ActivityBaseHomeostasis(Behaviour):
    def set_variables(self, neurons):
        self.max_activity = self.get_init_attr("max_activity", 50, neurons)
        self.min_activity = self.get_init_attr("min_activity", 10, neurons)
        self.window_size = self.get_init_attr("window_size", 100, neurons)
        self.updating_rate = self.get_init_attr("updating_rate", 0.001, neurons)

        # TODO: unsafe code need more time to digest the possibilities
        activity_rate = self.get_init_attr("activity_rate", 5, neurons)
        if isinstance(activity_rate, float) or isinstance(activity_rate, int):
            activity_rate = neurons.get_neuron_vec(mode="ones") * activity_rate

        self.activity_step = -self.window_size / activity_rate
        self.activities = neurons.get_neuron_vec(mode="zeros")
        self.exhaustion = neurons.get_neuron_vec()

    def new_iteration(self, neurons):
        changes = np.where(neurons.fired, 1, self.activity_step)
        self.activities += changes

        if (neurons.iteration % self.window_size) == 0:
            greater = ((self.activities > self.max_activity) * -1).astype(def_dtype)
            smaller = ((self.activities < self.min_activity) * 1).astype(def_dtype)
            greater *= self.activities - self.max_activity
            smaller *= self.max_activity - self.activities

            change = (greater + smaller) * self.updating_rate
            self.exhaustion += change
            neurons.threshold += self.exhaustion
