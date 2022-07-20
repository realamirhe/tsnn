import numpy as np

from PymoNNto import Behaviour

# should be after or be
from src.configs.plotters import pos_threshold_plotter, neg_threshold_plotter
from src.core.environement.homeostasis import HomeostasisEnvironment


class PopulationActivityBaseHomeostasis(Behaviour):
    def set_variables(self, n):
        self.updating_rate = self.get_init_attr("updating_rate", 0.001, n)
        self.activities = n.get_neuron_vec(mode="zeros")
        # self.phase = PhaseDetectorEnvironment.phase
        # self.threshold_phase = None
        # self.inference_start_threshold = n.threshold.copy()

    def new_iteration(self, n):
        # if self.threshold_phase != PhaseDetectorEnvironment.phase:
        #     new_threshold_phase = PhaseDetectorEnvironment.phase
        #     if new_threshold_phase == PhaseDetectorEnvironment.learning:
        #         n.threshold = self.inference_start_threshold
        #     else:
        #         # capture the first threshold
        #         self.inference_start_threshold = n.threshold.copy()
        #     self.threshold_phase = PhaseDetectorEnvironment.phase
        #
        # if self.phase != PhaseDetectorEnvironment.phase:
        #     self.reset()
        #     self.phase = PhaseDetectorEnvironment.phase
        if "pos" in n.tags:
            pos_threshold_plotter.add(n.threshold)
        else:
            neg_threshold_plotter.add(n.threshold)

        if not HomeostasisEnvironment.has_enough_data:
            return

        self.activities += n.fired.astype(int)

        if np.max(self.activities) > 10:
            debug = "activities"
        firing_reward = 1
        # per sentences
        neuron_activity_per_sentence = n.base_activity

        if HomeostasisEnvironment.enabled:  # should be enabled after 4 sentence
            # smoothing add one method
            change = self.activities - (neuron_activity_per_sentence or 1.0)
            change *= self.updating_rate
            n.threshold += change
            self.activities *= 0
