import numpy as np

from PymoNNto import Behaviour

# should be after or be
from src.configs.plotters import activity_plotter, threshold_plotter, dst_firing_plotter


class ActivityBaseHomeostasis(Behaviour):
    def set_variables(self, n):
        self.window_size = self.get_init_attr("window_size", 100, n)
        self.updating_rate = self.get_init_attr("updating_rate", 0.001, n)

        # activity_rate = np.ceil(self.get_init_attr("activity_rate", 5, n) / n.size)
        activity_rate = self.get_init_attr("activity_rate", 5, n) / n.size
        # NOTE: it might cause an error in the long them
        if activity_rate * n.size > self.window_size:
            print(f"{activity_rate=} {n.size=} {self.window_size=}")
            raise Exception(
                "Ceiling the activity in this window size cause problem in homeostasis"
            )

        if not isinstance(activity_rate, (float, int)):
            raise Exception(
                "Activity rate is total accumulate desired activity rate for all neurons in the window size"
                + " and must be int or float"
            )

        self.firing_reward = 1
        self.non_firing_penalty = -activity_rate / (self.window_size - activity_rate)

        self.activities = n.get_neuron_vec(mode="zeros")
        self.exhaustion = n.get_neuron_vec(mode="zeros")
        # self.history = []
        # self.counter = Counter()

    def new_iteration(self, n):
        self.activities += np.where(
            n.fired,
            self.firing_reward,
            self.non_firing_penalty,
        )
        activity_plotter.add(self.activities)
        dst_firing_plotter.add(n.fired)
        # self.history.append(n.fired.copy())
        if (n.iteration % self.window_size) == 0:
            self.activities[np.isclose(self.activities, 0)] = 0
            # b = 1 << np.arange(self.history[0].size)
            # count = Counter([c.dot(b) for c in self.history])
            # self.counter.update(count)
            # print("[counter]", self.counter)
            change = (
                -self.activities
                * self.updating_rate
                * 0.99 ** (n.iteration // self.window_size)
            )
            # self.history.clear()

            # For: Logic for adaptive updating rate (see old trunks)
            n.threshold -= change
            threshold_plotter.add(n.threshold)
            # TODO: normalize the activity
            self.activities *= 0
