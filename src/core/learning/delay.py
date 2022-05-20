import numpy as np

from PymoNNto import Behaviour
from src.data.feature_flags import magically_hardcode_the_delays
from src.data.plotters import delay_plotter, selected_delay_plotter


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

        if magically_hardcode_the_delays:
            synapse.delay = (
                np.random.random((depth_size, synapse.src.size)) * self.max_delay + 1
            )
            synapse.delay[0, [0, 1, 2]] = [3, 2, 1]
            synapse.delay[1, [14, 12, 13]] = [3, 2, 1]

        """ History or neuron memory for storing the spiked activity over times """
        self.delayed_spikes = np.zeros(
            (depth_size, synapse.src.size, self.max_delay), dtype=bool
        )
        self.weight_share = np.zeros_like(self.delayed_spikes, dtype=np.float32)

        self.update_delay_float(synapse)

    # NOTE: delay behaviour only update internal vars corresponding to delta delay update.
    def new_iteration(self, synapse):
        """
            1. @[update_delay_float]
            2. copy src.fired into the new_spike
            3. get delayed spikes for current time step
            4. bypas (immediate-spike) the new_spike if the synapse.delay is zero (see eq `@t_spikes` in notion)
            5. override the synapse.src.fired (see `#note: max_tspike`)
            6. roll the delayed spikes
            7. insert the new_spike into the delayed spikes via existing mask
            8. update the weight_share

        @note:
            max_tspike:   Imagine the following values in the final layer
            abc:[a, b, c] = T // seen chars were abc + delay = [3,2,1]
            omn:[] = T // seen chars were abc + delay = [2,2,1]
            so we expect that the output fired character from the source layer to be [a,b,c]
            despite the fact that they haven't been fired in the omn layer 🐳
            As a quote "neurons activity is based on one of its own delayed activity"
        """

        self.update_delay_float(synapse)
        new_spikes = synapse.src.fired.copy()
        """ Spike immediately for neurons with zero delay """
        t_spikes = self.delayed_spikes[:, :, -1]
        # NOTE: supress where in case there is no action to be done!
        t_spikes = np.where(
            self.int_delay == 0,
            new_spikes[np.newaxis, :] * np.ones_like(t_spikes),
            t_spikes,
        )
        synapse.src.fired = np.max(t_spikes, axis=0)  # (see `#note: max_tspike`)

        """ Go ahead one time step (t+1), [shift right with zero] """
        self.delayed_spikes[:, :, -1] = 0
        self.delayed_spikes = np.roll(self.delayed_spikes, 1, axis=2)

        """" Insert newly received spikes to their latest delayed position """
        self.delayed_spikes = np.where(
            self.delay_mask,
            new_spikes[np.newaxis, :, np.newaxis] * np.ones_like(self.delayed_spikes),
            self.delayed_spikes,
        )

        synapse.weights_scale *= t_spikes
        delay_plotter.add_image(synapse.delay, vmin=0, vmax=self.max_delay)

    def update_delay_float(self, synapse):
        """
        @note:
            weight_share_shape:   Note that the delay is clamped between [0, max_delay]
            and we are getting floor for the int_delay (relative actual placement index) so the max delay
            will cause a placement of `1` in the index 0 of that connection and zero for the next timestep
            which can safely be ignored
        """
        # synapse.delay = np.clip(np.round(synapse.delay, 1), 0, self.max_delay)
        synapse.delay = np.clip(synapse.delay, 0, self.max_delay)
        selected_delay_plotter.add(
            np.concatenate(
                (synapse.delay[0, [0, 1, 2]], synapse.delay[1, [14, 13, 12]]), axis=0
            )
        )
        # print("delay", synapse.delay.flatten())
        """ int_delay: (src.size, dst.size) """
        self.int_delay = np.floor(synapse.delay).astype(dtype=int)
        """ update delay mask (dst.size, src.size, max_delay) """
        self.delay_mask = np.zeros_like(self.delayed_spikes, dtype=bool)
        for dst_idx in range(self.int_delay.shape[0]):
            """ Set neurons in delay index to True """
            for delay, row in zip(self.int_delay[dst_idx], self.delay_mask[dst_idx]):
                if delay != 0:
                    row[-delay] = True

        # TODO: maybe move to another function make call predictable
        """ Update weight share based on float delays """
        synapse_delayed_effected_share = 1 - synapse.delay[:, :, np.newaxis] % 1.0
        synapse_delayed_effected_share[synapse_delayed_effected_share == 0] = 1.0
        synapse_delayed_effected_share = (
            synapse_delayed_effected_share
            * np.ones_like(self.weight_share)
            * self.delay_mask
        )

        """ accumulative update of weight_share """
        weight_share_update = (
            synapse_delayed_effected_share * self.delay_mask
            + np.roll(
                np.round((1 - synapse_delayed_effected_share) * self.delay_mask, 1),
                -1,
                axis=2,
            )
        )

        synapse.weights_scale = self.weight_share[:, :, -1].copy()
        self.weight_share[:, :, -1] = 0
        self.weight_share = np.roll(self.weight_share, 1, axis=2)
        self.weight_share += weight_share_update
