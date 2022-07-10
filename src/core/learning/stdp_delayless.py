import numpy as np

from PymoNNto import Behaviour
from src.core.environement.dopamine import DopamineEnvironment
from src.core.learning.stdp import bounds


class SynapsePairWiseSTDPWithoutDelay(Behaviour):
    __slots__ = [
        "a_minus",
        "a_plus",
        "dt",
        "stdp_factor",
        "tau_minus",
        "tau_plus",
        "w_max",
        "w_min",
        "weight_update_strategy",
    ]

    def set_variables(self, synapse):

        configure = {
            "a_minus": -0.1,
            "a_plus": 0.2,
            "dt": 1.0,
            "stdp_factor": 1.0,
            "tau_minus": 3.0,
            "tau_plus": 3.0,
            "tau_pop": 3.0,
            "w_max": 10.0,
            "w_min": 0.0,
            "P": 1,
            "is_inhibitory": False,
            "weight_update_strategy": None,
        }

        for attr, value in configure.items():
            setattr(self, attr, self.get_init_attr(attr, value, synapse))

        synapse.J = self.get_init_attr("J", 10, synapse)
        synapse.W = synapse.get_synapse_mat("zeros")
        synapse.W = np.random.normal(
            loc=synapse.J / synapse.src.size,  # pre-synaptic
            scale=1 * self.P,
            size=synapse.W.shape,  # dst, src
        )
        synapse.W = np.clip(synapse.W, self.w_min, self.w_max)

        if self.a_minus >= 0:
            raise AssertionError("a_minus should be negative")

        if self.weight_update_strategy not in (None, "soft-bound", "hard-bound"):
            raise AssertionError(
                "weight_update_strategy must be one of soft-bound|hard-bound|None"
            )

    def new_iteration(self, synapse):
        # For testing only, we won't update synapse weights in test mode!
        if not synapse.recording:
            return

        synapse.src.trace[:, 0] += (
            -synapse.src.trace[:, 0] / self.tau_plus + synapse.src.fired  # dx
        ) * self.dt

        synapse.dst.trace[:, 0] += (
            -synapse.dst.trace[:, 0] / self.tau_minus + synapse.dst.fired  # dy
        ) * self.dt

        # np.sum(synapse.src.A_history[0] * synapse.src.size)
        # np.sum(synapse.src.fired) -> activate so soon
        synapse.dst.alpha[0] = synapse.dst.alpha[0] / self.tau_pop + (
            (1, -1)[self.is_inhibitory] * synapse.src.A_history[0]
        )

        ltd = synapse.src.fired * synapse.dst.trace[:, 0][:, np.newaxis]
        ltp = synapse.src.trace[:, 0] * synapse.dst.fired[:, np.newaxis]

        # soft bound for both delay and stdp separate
        dw = (
            DopamineEnvironment.get()  # from global environment
            * (self.a_plus * ltp + self.a_minus * ltd)  # stdp mechanism
            * bounds[self.weight_update_strategy or "none"](
                self.w_min, synapse.W, self.w_max
            )
            * self.stdp_factor  # stdp scale factor
            * synapse.enabled  # activation of synapse itself
            * self.dt
        )

        synapse.W[synapse.W > 0.01] -= 1e-5
        synapse.W = synapse.W + dw
        synapse.W = np.clip(synapse.W, self.w_min, self.w_max)
