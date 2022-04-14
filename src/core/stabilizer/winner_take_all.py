import numpy as np

from PymoNNto import Behaviour


class WinnerTakeAll(Behaviour):
    # def set_variables(self, n):
    #     assert (
    #         n.old_v
    #     ), "ng group must have old v config, please add `capture_old_v=True` to your StreamableLIFNeurons"

    def new_iteration(self, n):
        fired = n.fired

        if np.sum(fired) > 1:
            temp_fired = n.get_neuron_vec(mode="zeros") > 0
            temp_fired[np.argmax(np.abs(n.old_v) * fired)] = True
            n.fired = temp_fired

        # testing purposes
        # assert np.sum(n.fired) <= 1, "More than one neuron fired"
