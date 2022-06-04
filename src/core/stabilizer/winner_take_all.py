import numpy as np

from PymoNNto import Behaviour


class WinnerTakeAll(Behaviour):
    def new_iteration(self, n):
        fired = n.fired
        if np.sum(fired) > 1:
            temp_fired = n.get_neuron_vec(mode="zeros") > 0
            """ NOTE: old_v can be negative, positive, or zero
                the true action is to select among the fired neurons only
                so we set the non fired neurons to negative-infinity
                and select the maximum index, the index would definitely be beside the fired ones. 
                NOTE: old_v will be reset to a brand new copy of v in the next iteration
            """
            n.old_v[np.logical_not(fired)] = np.NINF
            temp_fired[np.argmax(n.old_v)] = True
            n.fired = temp_fired
