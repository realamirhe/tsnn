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

    def set_variables(self, n):
        configure = {
            "noise_scale_factor": 1,
            "adaptive_noise_scale": 1,
            "synapse_lens_selector": [{"path": ["All", 0], "type": "delayed"}],
            "stimulus_scale_factor": 1,
        }

        for attr, value in configure.items():
            setattr(self, attr, self.get_init_attr(attr, value, n))

    def new_iteration(self, n):
        # NOTE: 🔥 Synapse selection make it specific to first synapse, which might not be proper in bigger network
        synapses = n.afferent_synapses
        next_layer_stimulus = np.zeros(n.I.shape)

        for connection in self.synapse_lens_selector:
            synapse = synapses
            for lens in connection["path"]:
                synapse = synapse[lens]
            if connection["type"] == "delayed":
                I_ext = np.sum(synapse.W * synapse.src_fire_effect, axis=1)
            else:
                I_ext = np.sum(synapse.W * synapse.src.fired, axis=1)
            # print(
            #     f"""
            # W ≈ {np.average(synapse.W)}
            # J = {getattr(synapse, 'J', None)}
            # P = 0.7
            # -------------
            # W´ = {np.sum(synapse.W != 0) / synapse.W.size * 100}
            # W ∈ [-7, 0]
            # """
            # )
            next_layer_stimulus += I_ext

        # shrink the noise scale factor at the beginning of each episode
        if synapse.iteration == 1:
            self.noise_scale_factor *= self.adaptive_noise_scale

        noise = (
            self.noise_scale_factor
            * (np.random.random(next_layer_stimulus.shape) - 0.5)
            * 2
        )
        n.I = next_layer_stimulus * self.stimulus_scale_factor + noise
        if "pos" in n.tags:
            words_stimulus_plotter.add(n.I)
