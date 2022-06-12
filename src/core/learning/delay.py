import numpy as np

from PymoNNto import Behaviour
from src.configs import feature_flags, corpus_config
from src.configs.plotters import selected_delay_plotter
from src.helpers.base import selected_neurons_from_words


class SynapseDelay(Behaviour):
    # fmt: off
    __slots__ = ["max_delay", "delayed_spikes", "weight_effect", "delay_mask"]

    # fmt: on
    def set_variables(self, synapse):
        self.max_delay = self.get_init_attr("max_delay", 0.0, synapse)
        use_shared_weights = self.get_init_attr("use_shared_weights", False, synapse)
        mode = self.get_init_attr("mode", "random", synapse)
        depth_size = 1 if use_shared_weights else synapse.dst.size

        if isinstance(mode, float):
            if mode == 0:
                raise AssertionError("mode can not be zero")
            synapse.delay = np.ones((depth_size, synapse.src.size)) * mode
        else:
            # Delay are initialized very high at our nervous system,
            # Hence we start with the N(max, max/2)
            deviation = self.max_delay / 2
            synapse.delay = np.random.normal(
                loc=self.max_delay,
                scale=deviation,
                size=(depth_size, synapse.src.size),
            )
            synapse.delay = np.clip(synapse.delay, deviation, self.max_delay)

        if feature_flags.enable_magic_delays:
            for i, word in enumerate(corpus_config.words):
                synapse.delay[
                    i, [corpus_config.letters.index(char) for char in word]
                ] = np.arange(self.max_delay, 0, -self.max_delay / len(word))

        """ History or neuron fire spike pattern over times """
        self.fired_history = np.zeros(
            (synapse.src.size, self.max_delay + 1), dtype=np.float32  # sparse matrix
        )
        self.src_fired_indices = np.mgrid[0 : synapse.src.size, 0 : synapse.src.size]
        self.src_fired_indices = self.src_fired_indices[1][: synapse.dst.size, :]

    # NOTE: delay behaviour only update internal vars corresponding to delta delay update.
    def new_iteration(self, synapse):
        """https://docs.google.com/spreadsheets/d/11Z07E7FCriw9YbbzBBVYK3270W9jz182chuYbtB6Lcw/edit#gid=0"""
        synapse.delay += 1e-5
        synapse.delay = np.clip(synapse.delay, 0, self.max_delay)
        rows, cols = selected_neurons_from_words()
        selected_delay_plotter.add(synapse.delay[rows, cols])

        self.fired_history = np.roll(self.fired_history, 1, axis=1)
        self.fired_history[:, 0] = synapse.src.fired

        delays_indices = synapse.delay.astype(int)
        mantis = synapse.delay % 1.0
        mantis[mantis == 0] = 1.0
        complement = 1 - mantis

        synapse.src.fire_effect = (
            self.fired_history[self.src_fired_indices, delays_indices] * complement
            + self.fired_history[
                self.src_fired_indices, np.clip(delays_indices + 1, 0, self.max_delay)
            ]
            * mantis
        )
