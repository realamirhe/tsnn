import numpy as np

from PymoNNto import Behaviour
from src.data.feature_flags import magically_hardcode_the_delays
from src.data.plotters import selected_delay_plotter


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
        self.weight_share = np.zeros(
            (depth_size, synapse.src.size, self.max_delay), dtype=np.float32
        )

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

            weight_share_shape:   Note that the delay is clamped between [0, max_delay]
            and we are getting floor for the int_delay (relative actual placement index) so the max delay
            will cause a placement of `1` in the index 0 of that connection and zero for the next timestep
            which can safely be ignored 🥸
        """
        synapse.delay = np.clip(synapse.delay, 0, self.max_delay)
        selected_delay_plotter.add(
            np.concatenate(
                (synapse.delay[0, [0, 1, 2]], synapse.delay[1, [14, 13, 12]]), axis=0
            )
        )

        new_spikes = synapse.src.fired

        # TODO: this can be remove in a direct add mechanism
        by_pass_spikes = np.zeros_like(new_spikes, dtype=float)

        for dst_index in range(self.weight_share.shape[0]):
            for (src_index,) in np.argwhere(new_spikes != 0):
                delay = synapse.delay[dst_index, src_index]
                delay_index = np.floor(delay).astype(dtype=int)
                mantis = delay % 1.0 or 1.0
                complement = 1 - mantis

                # Delay is going to bypass major section of itself to the current spikes output
                if delay_index == 0:
                    by_pass_spikes[src_index] = mantis
                    self.weight_share[dst_index, src_index, -1] += complement
                elif delay_index == self.max_delay:
                    self.weight_share[dst_index, src_index, -delay_index] += mantis
                else:
                    self.weight_share[dst_index, src_index, -delay_index] += mantis
                    self.weight_share[
                        dst_index, src_index, -delay_index - 1
                    ] += complement

        # TODO: cause we are transferring effect avg and sum might be valid actions too
        synapse.src.fired = np.max(self.weight_share[:, :, -1], axis=0) + by_pass_spikes
        # convert float value to spike
        synapse.src.fired = synapse.src.fired != 0
        synapse.weights_scale = self.weight_share[:, :, -1].copy()
        self.weight_share[:, :, -1] = 0
        self.weight_share = np.roll(self.weight_share, 1, axis=2)
