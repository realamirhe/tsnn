import numpy as np

from PymoNNto import Behaviour

# should be after or be
from src.configs.plotters import (
    activity_plotter,
    pos_threshold_plotter,
    neg_threshold_plotter,
)
from src.core.environement.inferencer import PhaseDetectorEnvironment


class ActivityBaseHomeostasis(Behaviour):
    def set_variables(self, n):
        self.window_size = self.get_init_attr("window_size", 100, n)
        self.updating_rate = self.get_init_attr("updating_rate", 0.001, n)

        # activity_rate = np.ceil(self.get_init_attr("activity_rate", 5, n) / n.size)
        activity_rate = self.get_init_attr("activity_rate", 5, n) / n.size
        # NOTE: it might cause an error in the long them
        # if activity_rate * n.size > self.window_size:
        #     raise Exception(
        #         "Ceiling the activity in this window size cause problem in homeostasis"
        #     )

        if not isinstance(activity_rate, (float, int)):
            raise Exception(
                "Activity rate is total accumulate desired activity rate for all neurons in the window size"
                + " and must be int or float"
            )

        self.firing_reward = 1
        self.non_firing_penalty = -activity_rate / (self.window_size - activity_rate)

        self.activities = n.get_neuron_vec(mode="zeros")
        self.counter = {"pos": 0, "neg": 0}
        self.counter_fired = {
            "pos": np.zeros_like(n.threshold),
            "neg": np.zeros_like(n.threshold),
        }
        self.phase = PhaseDetectorEnvironment.phase
        self.threshold_phase = None
        self.inference_start_threshold = n.threshold.copy()

    # @deprecated action, please use the ConditionalReset
    def reset(self):
        self.counter = {key: 0 for key in self.counter}
        self.activities *= 0

    def new_iteration(self, n):
        if self.threshold_phase != PhaseDetectorEnvironment.phase:
            new_threshold_phase = PhaseDetectorEnvironment.phase
            if new_threshold_phase == PhaseDetectorEnvironment.learning:
                n.threshold = self.inference_start_threshold
            else:
                # capture the first threshold
                self.inference_start_threshold = n.threshold.copy()
            self.threshold_phase = PhaseDetectorEnvironment.phase

        if self.phase != PhaseDetectorEnvironment.phase:
            self.reset()
            self.phase = PhaseDetectorEnvironment.phase

        self.counter[n.tags[0]] += np.sum(n.fired)
        self.counter_fired[n.tags[0]] += n.fired

        self.activities += np.where(
            n.fired,
            self.firing_reward,
            self.non_firing_penalty,
        )

        if "pos" in n.tags:
            activity_plotter.add(self.activities)
        # dst_firing_plotter.add(n.fired)

        if (n.iteration % self.window_size) == 0:
            self.activities[np.isclose(self.activities, 0)] = 0
            change = (
                -self.activities
                * self.updating_rate
                # NOTE: this going to make a non-similar topological noise object
                # * 0.99 ** (n.iteration // self.window_size)
            )
            # For: Logic for adaptive updating rate (see old trunks)
            n.threshold -= change

            self.activities *= 0
            self.counter[n.tags[0]] *= 0
            self.counter_fired[n.tags[0]] *= 0

        if "pos" in n.tags:
            pos_threshold_plotter.add(n.threshold)
        else:
            neg_threshold_plotter.add(n.threshold)
