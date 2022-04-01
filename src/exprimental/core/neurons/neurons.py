import numpy as np

from PymoNNto import Behaviour


class LIFNeuron(Behaviour):
    __slots__ = ["stream", "dt"]

    def set_variables(self, n):
        # NOTE: this is best for scope privilege principle but make code more difficult to config
        self.dt = self.get_init_attr("dt", 0.1, n)
        self.stream = self.get_init_attr("stream", None, n)

        configure = {
            "v_rest": -65,
            "v_reset": -65,
            "threshold": -52,
            "R": 1,
            "tau": 3,
            "v": n.get_neuron_vec(mode="ones"),
            "fired": n.get_neuron_vec(mode="zeros") > 0,
            "I": n.get_neuron_vec(mode="zeros"),
        }

        for attr, value in configure.items():
            setattr(n, attr, self.get_init_attr(attr, value, n))
        n.v *= n.v_rest

    def new_iteration(self, n):
        # TODO: must be added to the stream itself
        n.I = self.stream[n.iteration - 1]
        dv_dt = (n.v_rest - n.v) + n.R * n.I
        n.v += dv_dt * self.dt / n.tau
        n.fired = n.v >= n.threshold
        if np.sum(n.fired) > 0:
            n.v[n.fired] = n.v_reset
