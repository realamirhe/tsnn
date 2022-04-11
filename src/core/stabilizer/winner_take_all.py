import numpy as np

from PymoNNto import Behaviour


class WinnerTakeAll(Behaviour):
    # def set_variables(self, neurons):
    #     assert (
    #         neurons.old_v
    #     ), "ng group must have old v config, please add `capture_old_v=True` to your StreamableLIFNeurons"

    def new_iteration(self, neurons):
        fired = neurons.fired

        if np.sum(fired) > 1:
            temp_fired = neurons.get_neuron_vec(mode="zeros") > 0
            temp_fired[np.argmax(neurons.old_v * fired)] = True
            neurons.fired = temp_fired

        # testing purposes
        # assert np.sum(neurons.fired) <= 1, "More than one neuron fired"
