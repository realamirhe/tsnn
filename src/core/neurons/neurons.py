import numpy as np

from PymoNNto import Behaviour
from src.configs import corpus
from src.data.feature_flags import DEBUG_MODE


class StreamableLIFNeurons(Behaviour):
    __slots__ = ["stream", "dt"]

    def set_variables(self, n):
        # NOTE: this is best for scope privilege principle but make code more difficult to config
        self.dt = self.get_init_attr("dt", 0.1, n)
        self.stream = self.get_init_attr("stream", None, n)

        if DEBUG_MODE:
            self.joined_corpus = self.get_init_attr("joined_corpus", None, n)
            if self.joined_corpus is not None:
                n.seen_char = ""

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
        n.v = n.v_rest + n.get_neuron_vec(mode="uniform") * (n.threshold - n.v_reset)

        # NOTE: 🧬 For long term support, will be used in e.g. Homeostasis
        if has_long_term_effect:
            n.threshold = np.ones_like(n.v) * n.threshold

    def new_iteration(self, n):

        if DEBUG_MODE and self.joined_corpus is not None:
            n.seen_char += self.joined_corpus[n.iteration - 1]
            n.seen_char = n.seen_char[-corpus.gap :]

        is_forced_spike = self.stream is not None

        if is_forced_spike:
            n.I = self.stream[n.iteration - 1]

        dv_dt = n.v_rest - n.v + n.R * n.I
        n.v += dv_dt * self.dt / n.tau
        if self.capture_old_v:
            n.old_v = n.v.copy()

        if is_forced_spike:
            n.fired[:] = False
            n.fired[n.I.astype(bool)] = True
        else:
            n.fired = n.v >= n.threshold

        if np.sum(n.fired) > 0:
            n.v[n.fired] = n.v_reset
