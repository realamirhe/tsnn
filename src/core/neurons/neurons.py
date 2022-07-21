import numpy as np

from PymoNNto import Behaviour
from src.configs import network_config
from src.configs.plotters import pos_voltage_plotter, neg_voltage_plotter
from src.core.environement.inferencer import PhaseDetectorEnvironment


class StreamableLIFNeurons(Behaviour):
    def set_variables(self, n):
        # NOTE: this is best for scope privilege principle but make code more difficult to config
        self.dt = self.get_init_attr("dt", 0.1, n)
        self.stream = self.get_init_attr("stream", None, n)

        if network_config.is_debug_mode:
            self.joined_corpus = self.get_init_attr("joined_corpus", None, n)
            if self.joined_corpus is not None:
                n.seen_char = []

        self.capture_old_v = self.get_init_attr("capture_old_v", False, n)
        has_long_term_effect = self.get_init_attr("has_long_term_effect", False, n)

        configure = {
            "I": n.get_neuron_vec(mode="zeros"),
            "R": 1,
            "fired": n.get_neuron_vec(mode="zeros") > 0,
            "tau": 3,
            "threshold": -52,
            "v_reset": -65,
            "v_rest": -65,
        }

        for attr, value in configure.items():
            setattr(n, attr, self.get_init_attr(attr, value, n))

        # n.v = n.get_neuron_vec(mode="ones") * n.v_rest
        n.v = n.v_rest * n.get_neuron_vec(mode="ones")

        # NOTE: 🧬 For long term support, will be used in e.g. Homeostasis
        if has_long_term_effect:
            n.threshold = (
                np.ones_like(n.v) * n.threshold
                + (np.random.random(n.v.shape) - 0.5) * 2
            )
        self.phase = PhaseDetectorEnvironment.phase

    def new_iteration(self, n):
        if network_config.is_debug_mode and self.joined_corpus is not None:
            if self.phase != PhaseDetectorEnvironment.phase:
                self.phase = PhaseDetectorEnvironment.phase
                n.seen_char.clear()
            n.seen_char.append(self.joined_corpus[n.iteration - 1])
            # n.seen_char = n.seen_char[-corpus_config.words_capture_window_size :]

        is_forced_spike = self.stream is not None

        if is_forced_spike:
            n.I = self.stream[n.iteration - 1]

        if "pos" in n.tags:
            pos_voltage_plotter.add(n.v)
        elif "neg" in n.tags:
            neg_voltage_plotter.add(n.v)

        dv_dt = (n.v_rest - n.v) * self.dt / n.tau + n.R * n.I
        n.v += dv_dt
        if self.capture_old_v:
            n.old_v = n.v.copy()

        if is_forced_spike:
            n.fired[:] = False
            n.fired[n.I.astype(bool)] = True
        else:
            n.fired = n.v >= n.threshold

        if np.sum(n.fired) > 0:
            n.v[n.fired] = n.v_reset
