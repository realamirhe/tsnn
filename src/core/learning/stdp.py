import numpy as np

from PymoNNto import Behaviour
from src.configs import feature_flags, corpus_config
from src.configs.plotters import (
    dw_plotter,
    w_plotter,
    selected_dw_plotter,
    selected_weights_plotter,
)
from src.core.environement.dopamine import DopamineEnvironment
from src.helpers.base import selected_neurons_from_words


class SynapsePairWiseSTDP(Behaviour):
    __slots__ = [
        "a_minus",
        "a_plus",
        "dt",
        "min_delay_threshold",
        "stdp_factor",
        "tau_minus",
        "tau_plus",
        "w_max",
        "w_min",
        "weight_decay",
        "weight_update_strategy",
    ]

    def set_variables(self, synapse):
        synapse.W = synapse.get_synapse_mat("uniform")

        configure = {
            "a_minus": -0.2,
            "a_plus": 0.1,
            "delay_factor": 1.0,
            "dt": 1.0,
            "min_delay_threshold": 0.15,
            "stdp_factor": 1.0,
            "tau_minus": 3.0,
            "tau_plus": 3.0,
            "w_max": 10.0,
            "w_min": 0.0,
            "weight_decay": 1.0,
            "weight_update_strategy": None,
        }

        for attr, value in configure.items():
            setattr(self, attr, self.get_init_attr(attr, value, synapse))
        # Scale W from [0,1) to [w_min, w_max)
        synapse.W = synapse.W * (self.w_max - self.w_min) + self.w_min
        synapse.W = np.clip(synapse.W, self.w_min, self.w_max)

        if feature_flags.enable_magic_weights:
            for i, word in enumerate(corpus_config.words):
                indices = [corpus_config.letters.index(char) for char in word]
                synapse.W[:, indices] = self.w_min
                synapse.W[i, indices] = self.w_max

        if self.a_minus >= 0:
            raise AssertionError("a_minus should be negative")

        self.delay_domains = np.arange(synapse.src.size, dtype=int) * np.ones(
            (synapse.dst.size, 1), dtype=int
        )
        self.delay_ranges = -np.floor(synapse.delay).astype(int) - 1

        if self.weight_update_strategy not in (None, "soft-bound", "hard-bound"):
            raise AssertionError(
                "weight_update_strategy must be one of soft-bound|hard-bound|None"
            )

        # http://www.scholarpedia.org/article/Spike-timing_dependent_plasticity#Weight_dependence:_hard_bounds_and_soft_bounds
        if self.weight_update_strategy is None:
            self.A_minus = lambda w: self.a_minus
            self.A_plus = lambda w: self.a_plus
        elif self.weight_update_strategy == "soft-bound":
            # Soft bounds: for large weights, synaptic depression dominates over potentiation
            self.A_minus = lambda w: w * self.a_minus
            self.A_plus = lambda w: (self.w_max - w) * self.a_plus
        elif self.weight_update_strategy == "hard-bound":
            # Hard bounds: an update rule with fixed parameters a- and a+ is used until the bounds are reached
            self.A_minus = lambda w: np.heaviside(-w, 0) * self.a_minus
            self.A_plus = lambda w: np.heaviside(self.w_max - w, 0) * self.a_plus

        selected_weights_plotter.configure_plot(ylim=[self.w_min, self.w_max + 0.2])

    def new_iteration(self, synapse):
        # For testing only, we won't update synapse weights in test mode!
        if not synapse.recording:
            return

        # add new trace to existing src trace history
        # we don't have access to the latest src asar till here
        # should we accumulate it to the previous last layer trace or just replace that with the latest one
        synapse.src.trace[:, -1] += (
            -synapse.src.trace[:, -1] / self.tau_plus + synapse.src.fired  # dx
        ) * self.dt

        synapse.dst.trace[:, -1] += (
            -synapse.dst.trace[:, -1] / self.tau_minus + synapse.dst.fired  # dy
        ) * self.dt

        dw_minus = (
            self.A_minus(synapse.W)
            * synapse.src.fire_effect  # 2x26
            * synapse.dst.trace[:, -1][:, np.newaxis]  # 2x1
        )

        dw_neutral = (
            self.A_plus(synapse.W)
            * synapse.src.fire_effect
            * (synapse.dst.trace[:, -1] * synapse.dst.fired)[:, np.newaxis]
        )

        dw_plus = (
            self.A_plus(synapse.W)
            * synapse.src.trace[self.delay_domains, self.delay_ranges]
            * synapse.dst.fired[:, np.newaxis]
        )

        dw = (
            DopamineEnvironment.get()  # from global environment
            * (dw_plus + dw_neutral + dw_minus)  # stdp mechanism
            * self.stdp_factor  # stdp scale factor
            * synapse.enabled  # activation of synapse itself
            * self.dt
        )

        dw_plotter.add_image(dw * 1e5)
        rows, cols = selected_neurons_from_words()
        selected_dw_plotter.add(dw[rows, cols])
        synapse.W = synapse.W * self.weight_decay + dw
        synapse.W = np.clip(synapse.W, self.w_min, self.w_max)
        selected_weights_plotter.add(synapse.W[rows, cols])
        w_plotter.add_image(synapse.W, vmin=self.w_min, vmax=self.w_max)
        """ stop condition for delay learning """
        if not feature_flags.enable_delay_update_in_stdp:
            return

        use_shared_delay = dw.shape != synapse.delay.shape
        if use_shared_delay:
            dw = np.mean(dw, axis=0, keepdims=True)

        non_zero_dw = (dw != 0).astype(bool)
        if not non_zero_dw.any():
            return

        should_update = np.min(np.where(non_zero_dw, synapse.delay, np.inf), axis=1)
        should_update[should_update == np.inf] = 0
        should_update = should_update > self.min_delay_threshold
        if should_update.any():
            synapse.delay -= dw * should_update[:, np.newaxis] * self.delay_factor
            # NOTE: that np.floor doesn't use definition of "floor-towards-zero"
            self.delay_ranges = -np.floor(synapse.delay).astype(int) - 1
