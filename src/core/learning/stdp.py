import numpy as np

from PymoNNto import Behaviour
from src.core.environement.dopamine import DopamineEnvironment


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
            "stimulus_scale_factor": 1,
        }

        for attr, value in configure.items():
            setattr(self, attr, self.get_init_attr(attr, value, synapse))
        # Scale W from [0,1) to [w_min, w_max)
        synapse.W *= (self.w_max - self.w_min) + self.w_min
        synapse.W = np.clip(synapse.W, self.w_min, self.w_max)

        self.weight_decay = 1 - self.weight_decay

        assert self.a_minus < 0, "a_minus should be negative"

    def new_iteration(self, synapse):
        # For testing only, we won't update synapse weights in test mode!
        if not synapse.recording:
            synapse.dst.I = synapse.W.dot(synapse.src.fired)
            return

        # print("cool stuff")
        # print("seen_char", synapse.src.seen_char)
        # print("delay", synapse.delay[:, 23:])
        # print("weights_scale", synapse.weights_scale[:, 23:])
        # print("meta_delayed_spikes", synapse.meta_delayed_spikes[:, 23:])
        # print("meta_new_spikes", synapse.meta_new_spikes)
        # print("neuron_fired", synapse.src.fired)

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
            should_update = (
                np.min(synapse.delay[non_zero_dw]) > self.min_delay_threshold
            )
            if should_update:
                synapse.delay[non_zero_dw] -= dw[non_zero_dw] * self.delay_factor

        next_layer_stimulus = synapse.W.dot(synapse.src.fired)
        # TODO: need to investigate more for diagonal feature
        noise = np.random.random(next_layer_stimulus.shape)
        synapse.dst.I = self.stimulus_scale_factor * next_layer_stimulus + noise

    # NOTE: We might need the add clamping mechanism to the 'I' for the dst layer


# NOTE: clamping is better to be part of neurons itself
