import numpy as np

from PymoNNto import Behaviour
from src.configs.plotters import words_stimulus_plotter


class CurrentStimulus(Behaviour):
    """
    - a -> decay
    - ab -> decay
    - abc -> decay
    - abc_ -> validation_mechanism (reward|punishment)
    """

    __slots__ = ["dopamine_decay", "outputs"]

    def set_variables(self, neurons):
        configure = {
            "noise_scale_factor": 1,
            "adaptive_noise_scale": 1,
            "synapse_lens_selector": ["All", 0],
            "stimulus_scale_factor": 1,
        }

        for attr, value in configure.items():
            setattr(self, attr, self.get_init_attr(attr, value, neurons))

    def new_iteration(self, neurons):
        # NOTE: 🔥 Synapse selection make it specific to first synapse, which might not be proper in bigger network
        synapse = neurons.afferent_synapses
        for lens in self.synapse_lens_selector:
            synapse = synapse[lens]

        next_layer_stimulus = np.sum(synapse.W * synapse.src.fire_effect, axis=1)
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
