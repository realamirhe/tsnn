import numpy as np

from PymoNNto import Behaviour
from src.core.environement.dopamine import DopamineEnvironment


class SynapsePairWiseSTDP(Behaviour):
    # fmt: off
    __slots__ = ["tau_plus", "tau_minus", "a_plus", "a_minus", "dt", "weight_decay", "stdp_factor", "delay_epsilon",
                 "w_min", "w_max"]

    # fmt: on
    def set_variables(self, synapse):
        synapse.W = synapse.get_synapse_mat("uniform")
        synapse.src.trace = synapse.src.get_neuron_vec()
        synapse.dst.trace = synapse.dst.get_neuron_vec()

        configure = {
            "tau_plus": 3.0,
            "tau_minus": 3.0,
            "a_plus": 0.1,
            "a_minus": 0.2,
            "dt": 1.0,
            "weight_decay": 0.0,
            "stdp_factor": 1.0,
            "delay_epsilon": 0.15,
            "w_min": 0.0,
            "w_max": 10.0,
            "stimulus_scale_factor": 1e4,
        }

        for attr, value in configure.items():
            setattr(self, attr, self.get_init_attr(attr, value, synapse))
        self.weight_decay = 1 - self.weight_decay

    def new_iteration(self, synapse):
        if not synapse.recording:
            synapse.dst.I = synapse.W.dot(synapse.src.fired)
            return

        synapse.src.trace += (
            -synapse.src.trace / self.tau_plus + synapse.src.fired  # dx
        ) * self.dt

        synapse.dst.trace += (
            -synapse.dst.trace / self.tau_minus + synapse.dst.fired  # dy
        ) * self.dt

        dw_minus = (
            -self.a_minus
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
            * synapse.weights_scale[:, :, 0]  # weight scale based on the synapse delay
            * self.stdp_factor  # stdp scale factor
            * synapse.enabled  # activation of synapse itself
            * self.dt
        )

        synapse.W = synapse.W * self.weight_decay + dw
        synapse.W = np.clip(synapse.W, self.w_min, self.w_max)

        """ stop condition for delay learning """
        use_shared_delay = dw.shape != synapse.delay.shape
        if use_shared_delay:
            dw = np.mean(dw, axis=0, keepdims=True)

        non_zero_dw = dw != 0
        if non_zero_dw.any():
            should_update = np.min(synapse.delay[non_zero_dw]) > self.delay_epsilon
            if should_update:
                synapse.delay[non_zero_dw] -= dw[non_zero_dw]

        next_layer_stimulus = synapse.W.dot(synapse.src.fired)
        # TODO: need to investigate more for diagonal feature
        synapse.dst.I = (
            np.random.random(next_layer_stimulus.shape) * self.stimulus_scale_factor
        ) + next_layer_stimulus


# NOTE: We might need the add clamping mechanism to the 'I' for the dst layer
# NOTE: clamping is better to be part of neurons itself
