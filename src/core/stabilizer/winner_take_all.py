import numpy as np

from PymoNNto import Behaviour


class WinnerTakeAll(Behaviour):
    __slots__ = ["old_v"]

    def set_variables(self, n):
        self.old_v = n.v

    def new_iteration(self, neurons):
        fired = neurons.fired

        if np.sum(fired) > 1:
            temp_fired = neurons.get_neuron_vec(mode="zeros") > 0
            temp_fired[np.argmax(self.old_v * fired)] = True
            neurons.fired = temp_fired

        # testing purposes
        # assert np.sum(neurons.fired) <= 1, "More than one neuron fired"
        self.old_v = neurons.v
