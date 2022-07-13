import numpy as np

from PymoNNto import Behaviour
from src.configs.plotters import words_stimulus_plotter


class PopCurrentStimulus(Behaviour):
    __slots__ = ["dopamine_decay", "outputs"]

    def set_variables(self, n):
        configure = {
            "noise_scale_factor": 1,
            "adaptive_noise_scale": 1,
            "synapse_lens_selector": ["All", 0],
            "stimulus_scale_factor": 1,
        }

        for attr, value in configure.items():
            setattr(self, attr, self.get_init_attr(attr, value, n))

    def new_iteration(self, n):
        # NOTE: 🔥 Synapse selection make it specific to first synapse, which might not be proper in bigger network
        synapse = n.afferent_synapses
        for lens in self.synapse_lens_selector:
            synapse = synapse[lens]

        # Population base connection stimulus (balance network)
        I_pop = 0.0
        for pre_synaptic_connection in n.afferent_synapses["All"]:
            if "words" not in " ".join(pre_synaptic_connection.tags):
                I_pop += pre_synaptic_connection.J * np.convolve(
                    pre_synaptic_connection.src.alpha,
                    pre_synaptic_connection.src.A_history,
                    "valid",
                )

        next_layer_stimulus = np.sum(synapse.W * synapse.src_fire_effect, axis=1)
        # shrink the noise scale factor at the beginning of each episode
        if synapse.iteration == 1:
            self.noise_scale_factor *= self.adaptive_noise_scale

        noise = (
            self.noise_scale_factor
            * (np.random.random(next_layer_stimulus.shape) - 0.5)
            * 2
        )
        synapse.dst.I = next_layer_stimulus * self.stimulus_scale_factor + noise
        words_stimulus_plotter.add(synapse.dst.I)
