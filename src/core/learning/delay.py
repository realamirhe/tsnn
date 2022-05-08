import numpy as np

from PymoNNto import Behaviour
from src.data.plotters import delay_plotter


class SynapseDelay(Behaviour):
    # fmt: off
    __slots__ = ["max_delay", "delayed_spikes", "weight_share", "int_delay", "delay_mask"]

    # fmt: on
    def set_variables(self, synapse):
        self.max_delay = self.get_init_attr("max_delay", 0.0, synapse)
        use_shared_weights = self.get_init_attr("use_shared_weights", False, synapse)
        mode = self.get_init_attr("mode", "random", synapse)
        depth_size = 1 if use_shared_weights else synapse.dst.size

        if isinstance(mode, float):

            assert mode != 0, "mode can not be zero"
            synapse.delay = np.ones((depth_size, synapse.src.size)) * mode
        else:
            synapse.delay = (
                np.random.random((depth_size, synapse.src.size)) * self.max_delay + 1
            )

        # MAGIC Delay setters
        # synapse.delay = (
        #         np.random.random((depth_size, synapse.src.size)) * self.max_delay + 1
        # )
        # synapse.delay[0, [0, 1, 2]] = [3, 2, 1]
        # synapse.delay[1, [14, 12, 13]] = [3, 2, 1]
        """ History or neuron memory for storing the spiked activity over times """
        self.delayed_spikes = np.zeros(
            (depth_size, synapse.src.size, self.max_delay), dtype=bool
        )
        self.weight_share = np.ones((depth_size, synapse.src.size, 2), dtype=np.float32)
        self.weight_share[:, :, -1] = 0.0

        self.update_delay_float(synapse)

    def new_iteration(self, synapse):
        self.update_delay_float(synapse)
        new_spikes = synapse.src.fired.copy()
        """ TBD: neurons activity is based on one of its own delayed activity """
        """ Spike immediately for neurons with zero delay """
        t_spikes = self.delayed_spikes[:, :, -1]
        # NOTE: supress where in case there is no action to be done!
        t_spikes = np.where(
            self.int_delay == 0,
            new_spikes[np.newaxis, :] * np.ones_like(t_spikes),
            t_spikes,
        )
        synapse.src.fired = np.max(t_spikes, axis=0)

        """ Go ahead one time step (t+1), [shift right with zero] """
        self.delayed_spikes[:, :, -1] = 0
        self.delayed_spikes = np.roll(self.delayed_spikes, 1, axis=2)

        """" Insert newly received spikes to their latest delayed position """
        self.delayed_spikes = np.where(
            self.delay_mask,
            new_spikes[np.newaxis, :, np.newaxis] * np.ones_like(self.delayed_spikes),
            self.delayed_spikes,
        )

        """ Calculate weight share of exported spikes """
        weight_scale = t_spikes[:, :, np.newaxis] * self.weight_share
        if hasattr(synapse, "weight_scale"):
            """ accumulative shift of weight_share """
            weight_scale[:, :, 0] += synapse.weights_scale[:, :, -1]
        synapse.weights_scale = weight_scale
        delay_plotter.add_image(synapse.delay, vmin=0, vmax=self.max_delay)

    def update_delay_float(self, synapse):
        synapse.delay = np.clip(np.round(synapse.delay, 1), 0, self.max_delay)
        # print("delay", synapse.delay.flatten())
        """ int_delay: (src.size, dst.size) """
        self.int_delay = np.floor(synapse.delay).astype(dtype=int)
        """ update delay mask (dst.size, src.size, max_delay) """
        self.delay_mask = np.zeros_like(self.delayed_spikes, dtype=bool)
        for n_idx in range(self.int_delay.shape[0]):
            """ Set neurons in delay index to True """
            for delay, row in zip(self.int_delay[n_idx], self.delay_mask[n_idx]):
                if delay != 0:
                    row[-delay] = True

        # TODO: maybe move to another function make call predictable
        """ Update weight share based on float delays """
        self.weight_share[:, :, 0] = synapse.delay % 1.0
        """ Full weight share for integer delays """
        weight_share_in_time_t = self.weight_share[:, :, 0]
        weight_share_in_time_t[weight_share_in_time_t == 0] = 1.0
        self.weight_share[:, :, 0] = weight_share_in_time_t
        """ forward remaining weight to the next time step """
        self.weight_share[:, :, 1] = 1 - self.weight_share[:, :, 0]
