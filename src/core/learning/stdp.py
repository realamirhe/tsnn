import numpy as np

from PymoNNto import Behaviour
from src.core.environement.dopamine import DopamineEnvironment
from src.data.feature_flags import (
    magically_hardcode_the_weights,
    prevent_delay_update_in_stdp,
)
from src.data.plotters import dw_plotter, w_plotter, selected_dw_plotter


class SynapsePairWiseSTDP(Behaviour):
    __slots__ = [
        "tau_plus",
        "tau_minus",
        "a_plus",
        "a_minus",
        "dt",
        "weight_decay",
        "stdp_factor",
        "min_delay_threshold",
        "w_min",
        "w_max",
    ]

    def set_variables(self, synapse):
        synapse.W = synapse.get_synapse_mat("uniform")
        synapse.src.trace = synapse.src.get_neuron_vec()
        synapse.dst.trace = synapse.dst.get_neuron_vec()

        configure = {
            "tau_plus": 3.0,
            "tau_minus": 3.0,
            "a_plus": 0.1,
            "a_minus": -0.2,
            "dt": 1.0,
            "weight_decay": 0.0,
            "stdp_factor": 1.0,
            "delay_factor": 1.0,
            "min_delay_threshold": 0.15,
            "w_min": 0.0,
            "w_max": 10.0,
        }

        for attr, value in configure.items():
            setattr(self, attr, self.get_init_attr(attr, value, synapse))
        # Scale W from [0,1) to [w_min, w_max)
        synapse.W = synapse.W * (self.w_max - self.w_min) + self.w_min
        synapse.W = np.clip(synapse.W, self.w_min, self.w_max)

        if magically_hardcode_the_weights:
            synapse.W[0, [0, 1, 2]] = self.w_max
            synapse.W[1, [0, 1, 2]] = self.w_min
            synapse.W[1, [12, 13, 14]] = self.w_max
            synapse.W[0, [12, 13, 14]] = self.w_min

        self.weight_decay = 1 - self.weight_decay

        assert self.a_minus < 0, "a_minus should be negative"

    def new_iteration(self, synapse):
        # For testing only, we won't update synapse weights in test mode!
        if not synapse.recording:
            return

        synapse.src.trace += (
            -synapse.src.trace / self.tau_plus + synapse.src.fired  # dx
        ) * self.dt

        synapse.dst.trace += (
            -synapse.dst.trace / self.tau_minus + synapse.dst.fired  # dy
        ) * self.dt

        dw_minus = (
            self.a_minus
            * synapse.src.fired[np.newaxis, :]
            * synapse.dst.trace[:, np.newaxis]
        )
        dw_plus = (
            self.a_plus
            * synapse.src.trace[np.newaxis, :]
            * synapse.dst.fired[:, np.newaxis]
        )

        dw = (
            DopamineEnvironment.get()  # from global environment
            * (dw_plus + dw_minus)  # stdp mechanism
            * synapse.weights_scale  # weight scale based on the synapse delay
            * self.stdp_factor  # stdp scale factor
            * synapse.enabled  # activation of synapse itself
            * self.dt
        )

        dw_plotter.add_image(dw * 1e5)
        selected_dw_plotter.add(
            np.concatenate((dw[0, [0, 1, 2]], dw[1, [14, 13, 12]]), axis=0)
        )
        synapse.W = synapse.W * self.weight_decay + dw
        synapse.W = np.clip(synapse.W, self.w_min, self.w_max)
        w_plotter.add_image(synapse.W, vmin=self.w_min, vmax=self.w_max)
        """ stop condition for delay learning """
        if prevent_delay_update_in_stdp:
            return

        use_shared_delay = dw.shape != synapse.delay.shape
        if use_shared_delay:
            dw = np.mean(dw, axis=0, keepdims=True)

        non_zero_dw = (dw != 0).astype(bool)
        if not non_zero_dw.any():
            return

        should_update = np.min(synapse.delay[non_zero_dw]) > self.min_delay_threshold
        if should_update:
            synapse.delay[non_zero_dw] -= dw[non_zero_dw] * self.delay_factor
