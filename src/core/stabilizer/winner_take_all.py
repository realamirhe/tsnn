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
            """ NOTE: old_v can be negative, positive, or zero
                the true action is to select among the fired neurons only
                so we set the non fired neurons to negative-infinity
                and select the maximum index, the index would definitely be beside the fired ones. 
            """
            n.old_v[np.logical_not(fired)] = np.NINF
            temp_fired[np.argmax(n.old_v)] = True
            n.fired = temp_fired

        # testing purposes
        # assert np.sum(n.fired) <= 1, "More than one neuron fired"
