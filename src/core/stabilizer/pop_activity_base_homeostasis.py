from PymoNNto import Behaviour

from src.configs.plotters import pos_threshold_plotter, neg_threshold_plotter
from src.core.environement.homeostasis import HomeostasisEnvironment
from src.core.environement.inferencer import PhaseDetectorEnvironment


class PopulationActivityBaseHomeostasis(Behaviour):
    def set_variables(self, n):
        self.updating_rate = self.get_init_attr("updating_rate", 0.001, n)
        self.activities = n.get_neuron_vec(mode="zeros")
        self.phase = PhaseDetectorEnvironment.phase
        self.threshold_phase = None
        self.inference_start_threshold = n.threshold.copy()

    def new_iteration(self, n):
        if "pos" in n.tags:
            pos_threshold_plotter.add(n.threshold)
        else:
            neg_threshold_plotter.add(n.threshold)

        if not HomeostasisEnvironment.has_enough_data:
            return

        if self.threshold_phase != PhaseDetectorEnvironment.phase:
            if PhaseDetectorEnvironment.is_phase("inference"):
                # capture the first threshold
                self.inference_start_threshold = n.threshold.copy()
            else:
                n.threshold = self.inference_start_threshold
            self.threshold_phase = PhaseDetectorEnvironment.phase

        self.activities += n.fired.astype(int)

        if HomeostasisEnvironment.enabled:  # should be enabled after 4 sentence
            # smoothing add one method
            neuron_activity_per_sentence = n.base_activity
            change = self.activities - (neuron_activity_per_sentence or 1.0)
            change *= self.updating_rate
            n.threshold += change
            self.activities *= 0
