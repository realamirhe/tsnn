import numpy as np

from PymoNNto import Behaviour
from src.config import feature_flags
from src.data.plotters import selected_delay_plotter


class SynapseDelay(Behaviour):
    # fmt: off
    __slots__ = ["max_delay", "delayed_spikes", "weight_effect", "int_delay", "delay_mask"]

    # fmt: on
    def set_variables(self, synapse):
        self.max_delay = self.get_init_attr("max_delay", 0.0, synapse)
        use_shared_weights = self.get_init_attr("use_shared_weights", False, synapse)
        mode = self.get_init_attr("mode", "random", synapse)
        depth_size = 1 if use_shared_weights else synapse.dst.size

        if isinstance(mode, float):
            if mode == 0:
                raise AssertionError("mode can not be zero")
            synapse.delay = np.ones((depth_size, synapse.src.size)) * mode
        else:
            # Delay are initialized very high at our nervous system, so we start with the N(max, max/2)
            deviation = self.max_delay / 2
            synapse.delay = np.random.normal(
                loc=self.max_delay,
                scale=deviation,
                size=(depth_size, synapse.src.size),
            )
            synapse.delay = np.clip(synapse.delay, deviation, self.max_delay)

        if feature_flags.enable_magic_delays:
            synapse.delay = (
                np.random.random((depth_size, synapse.src.size)) * self.max_delay + 1
            )
            synapse.delay[0, [0, 1, 2]] = [3, 2, 1]
            synapse.delay[1, [14, 12, 13]] = [3, 2, 1]

        """ History or neuron memory for storing the spiked activity over times """
        self.weight_effect = np.zeros(
            (depth_size, synapse.src.size, self.max_delay + 1), dtype=np.float32
        )

    # NOTE: delay behaviour only update internal vars corresponding to delta delay update.
    def new_iteration(self, synapse):
        """
            1. clip the synapse delay between its boundary
            2. for every dst layer connection as `dst_index` do
                2.1. for every spiked src 🚀 layer connection as `src_index` do
                    2.1.1. update weight share and by_pass connection (in zero delay) directly on the existing variables
            3. calculate the synapse.src.fired based on the existing weight-share (@note weight_share_2_firing_pattern)
            4. convert floating nonzero effect to boolean spike pattern for synapse.src.fired
            5. get copy of the active weight share
            6. make the forward step in the time (zero last + roll)

        https://docs.google.com/spreadsheets/d/11Z07E7FCriw9YbbzBBVYK3270W9jz182chuYbtB6Lcw/edit#gid=0
        @note:
            weight_share_2_firing_pattern:    Every connection in the `t` layer is caused by previous seen character
            (input) in the previous layer so, it only co-occurrence which might cause an issue is when neurons in the
            output layer has same synapse delay (or larger) as the accumulated weight share become strong enough to
            activate next layer input. So keeping the `t` layer of all output neurons will give us `weight_scale`
            and the maximum of weight scale in the output axis will give us the firing pattern
        """
        synapse.delay = np.clip(synapse.delay, 0, self.max_delay)
        selected_delay_plotter.add(
            synapse.delay[[0, 0, 0, 1, 1, 1], [0, 1, 2, 14, 12, 13]]
        )

        new_spikes = synapse.src.fired
        activated_src_neurons = np.argwhere(new_spikes != 0)

        # NOTE: consider swap for loop and benchmarking performance 🐝
        for dst_index in range(self.weight_effect.shape[0]):
            for (src_index,) in activated_src_neurons:
                delay = synapse.delay[dst_index, src_index]
                delay_index = np.floor(delay).astype(dtype=int)
                """
                Intelligence switch for delay share (t=3, t=2) (complement, mantis)
                delay=2.3 => (0.7, 0.3) => should be **switched** it is closed to 2 most share are for 2 (base_t=2)
                delay=2.7 => (0.3, 0.7) => should be **switched** it is closed to 3 most share are for 3 (base_t=2)
                delay=3 => (1, 0) (base_t=3)
                delay=2 => (1, 0) (base_t=2)
                """
                mantis = delay % 1.0 or 1.0
                complement = 1 - mantis
                # Delay is going to bypass major section of itself to the current spikes output
                if delay_index == 0:
                    self.weight_effect[dst_index, src_index, -2:] += [
                        mantis,
                        complement,
                    ]
                # Do not do the complement update in case it will be zero
                elif complement == 0:
                    self.weight_effect[dst_index, src_index, -delay_index - 1] += mantis
                else:
                    self.weight_effect[
                        dst_index, src_index, -delay_index - 2 : -delay_index
                    ] += [mantis, complement]

        synapse.src.fire_effect = self.weight_effect[:, :, -1].copy()

        self.weight_effect[:, :, -1] = 0
        self.weight_effect = np.roll(self.weight_effect, 1, axis=2)
