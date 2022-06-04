import numpy as np
from matplotlib import pyplot as plt

from PymoNNto import Behaviour, SynapseGroup, Recorder, NeuronGroup, Network
from PymoNNto.Exploration.Network_UI import get_default_UI_modules, Network_UI
from src.trunk.libs import behaviours
from src.trunk.libs.data_generator_numpy import stream_generator_for_character
from src.trunk.libs.environment import set_dopamine, get_dopamine
from src.trunk.libs.helper import behaviour_generator

# =================   CONFIG    =================
UI_MODE = False
ITERATIONS = 50
i_stream = stream_generator_for_character("i", noise=0.4, size=ITERATIONS)
j_stream = stream_generator_for_character("j", noise=0.1, size=ITERATIONS)
streams = {"i": i_stream, "j": j_stream}
print({chr: "".join([i for i in stream]) for chr, stream in streams.items()})


# ================= BEHAVIOURS  =================
class LIFMechanism(Behaviour):
    def set_variables(self, n):
        self.add_tag("LIFMechanism")
        self.set_init_attrs_as_variables(n)
        self.with_firing_rate = self.get_init_attr("with_firing_rate", False, n)
        n.v = n.v_rest + n.get_neuron_vec("uniform") * 10  # voltage
        n.fired = n.get_neuron_vec("zeros") > 0
        n.dt = getattr(n, "dt", 0.1)

    def new_iteration(self, n):
        n.v += ((n.v_rest - n.v) + n.I) * n.dt

        if self.with_firing_rate:
            n.fired = n.v > n.v_threshold
            if np.sum(n.fired) > 0:
                n.v[n.fired] = n.v_reset


# NOTE: Test has not been managed yet
class Delay(Behaviour):
    """
    This module can be behavioursused to delay the input of a NeuronGroup.
    delayed_spikes => [t+max_delay, ...,t+2, t+1, t]
    delays => [0..max_delay]xN
    delays_float => float32 delays
    """

    def set_variables(self, n):
        self.add_tag("Delay")
        self.max_delay = self.get_init_attr("max_delay", 0, n)
        self.delayed_spikes = np.zeros((n.size, self.max_delay), dtype=bool)
        self.delays_float = np.zeros(n.size, dtype=np.float32)  # For delay learning
        self.delays = np.ceil(self.delays_float).astype(dtype=int)

        self.init_delays(n)
        self.update_delay_mask()

    # TODO: update mechanism for delay must also call the `self.update_delay_mask`
    # TODO: affect the shared weights in related synapse
    # TODO: delay update by post-synaptic neuron
    # TODO: weight sharing mechanism is simpler now :) see the delay mechanism
    def new_iteration(self, n):
        if self.max_delay == 0:
            return

        new_spikes = n.fired.copy()

        n.fired = self.delayed_spikes[:, -1].copy()
        # Spike immediately in zero delay
        n.fired[self.delays == 0] = new_spikes[self.delays == 0]

        # Go ahead one time step (t+1), [shift right with zero]
        self.delayed_spikes[:, -1] = 0
        self.delayed_spikes = np.roll(self.delayed_spikes, 1, axis=1)
        # Insert newly received spikes to their latest delayed position
        self.delayed_spikes = np.where(
            self.latest_spike_index,
            new_spikes[:, np.newaxis] * np.ones_like(self.delayed_spikes),
            self.delayed_spikes,
        )

        # TODO: add weight sharing mechanism
        # FIXME: NO hard code
        for s in n.efferent_synapses["GLUTAMATE"]:
            will_spike = np.sum(self.delayed_spikes, axis=1)
            will_spike[will_spike == 0] = 1
            s.weights_scale = 1 / will_spike

    def init_delays(self, n):
        self.delay_method = self.get_init_attr("delay_method", "random", n)
        if self.delay_method == "zero":
            self.delays = 0
            self.delays_float = 0

        elif self.delay_method == "random":
            self.delays = np.random.randint(0, self.max_delay + 1, n.size)
            self.delays_float = self.delays.astype(dtype=np.float32)

        elif self.delay_method == "constant":
            delay_constant = self.get_init_attr("delay_constant", 0, n)
            # It is the same shape array or constant int less than max delay
            assert (
                type(delay_constant) == np.ndarray and delay_constant.size == n.size
            ) or (type(delay_constant) == int and delay_constant <= self.max_delay)
            self.delays[:] = delay_constant
            self.delays_float = self.delays.astype(dtype=np.float32)

    def update_delay_mask(self):
        self.latest_spike_index = np.zeros_like(self.delayed_spikes, dtype=bool)
        for delay, row in zip(self.delays, self.latest_spike_index):
            if delay != 0:
                row[-delay] = True


# NOTE: size of neurons must be 1x1
class CharacterSpikeStream(Behaviour):
    def set_variables(self, n):
        self.size = self.get_init_attr("stream_size", 0, n)
        self.target_char = self.get_init_attr("target_char", None, n)
        self.stream = streams[self.target_char]

    def new_iteration(self, n):
        # TODO: noise in iteration stream spikes
        if np.random.random() > 0.85:
            return

        if not UI_MODE:
            n.fired[:] = self.stream[n.iteration - 1] == self.target_char
        else:
            n.fired[:] = self.stream[(n.iteration - 1) % self.size] == self.target_char

        if np.sum(n.fired) > 0:
            n.v[n.fired] = n.v_reset


class DopamineProvider(Behaviour):
    def set_variables(self, n):
        self.add_tag("Dopamine")
        # TODO: test for more 1increased verison and observe the difference
        self.dopamine_scale = self.get_init_attr("dopamine_scale", 0.9, n)

    def new_iteration(self, n):
        """Evaluate function for dopamine control"""
        if "j" not in n.tags:
            return

        if not UI_MODE:
            step_label = streams["j"][n.iteration - 1]
        else:
            step_label = streams["j"][(n.iteration - 1) % ITERATIONS]

        if n.fired[0]:
            if step_label != "j":
                print("dopamine ⤵")
                set_dopamine(-1)
            else:
                print("dopamine ⤴")
                set_dopamine(1)
        else:
            print("dopamine🔻")
            set_dopamine(get_dopamine() * self.dopamine_scale)


# ================= NETWORK  =================
def main():
    network = Network()
    neuron_i = NeuronGroup(
        net=network,
        tag="i",
        size=1,
        behaviour=behaviour_generator(
            [
                LIFMechanism(v_rest=-65, v_reset=-65, v_threshold=-52),
                # behaviours.ForceSpikeOnLetters(),
                CharacterSpikeStream(target_char="i", stream_size=ITERATIONS),
                behaviours.LIFInput(),
                # Delay(max_delay=3, delay_method="random"),
                # Delay(max_delay=3, delay_method="zero"),
                Delay(max_delay=5, delay_method="constant", delay_constant=3),
                behaviours.STDP(stdp_factor=0.00015),
                # behaviours.Normalization(norm_factor=10),
                Recorder(tag="IRecorder", variables=["n.v", "n.fired"]),
            ]
        ),
    )

    neuron_j = NeuronGroup(
        net=network,
        tag="j",
        size=1,
        behaviour=behaviour_generator(
            [
                LIFMechanism(
                    v_rest=-65, v_reset=-80, v_threshold=-52, with_firing_rate=True
                ),
                # CharacterSpikeStream(target_char="j", stream_size=ITERATIONS),
                behaviours.LIFInput(),
                behaviours.STDP(stdp_factor=0.00015),
                # behaviours.Homeostasis(
                #     target_voltage=0.05, max_ta=-30, min_ta=-60, eta_ip=0.0001
                # ),
                # behaviours.Normalization(norm_factor=10),
                DopamineProvider(dopamine_scale=0.001),
                Recorder(tag="JRecorder", variables=["n.v", "n.fired"]),
            ]
        ),
    )

    SynapseGroup(net=network, src=neuron_i, dst=neuron_j, tag="GLUTAMATE")
    # SynapseGroup(net=network, src=neuron_j, dst=neuron_j, tag="GABA")

    network.initialize()
    network.simulate_iterations(ITERATIONS, measure_block_time=True)

    if not UI_MODE:
        plt.plot(network["IRecorder", 0]["n.v", 0])
        plt.title("neuron(I).voltage (trace)")
        plt.xlabel("iterations")
        plt.ylabel("voltage")
        plt.show()

        plt.plot(network["JRecorder", 0]["n.v", 0])
        plt.title("neuron(J).voltage (trace)")
        plt.show()

        plt.imshow(
            network["IRecorder", 0]["n.fired", 0, "np"].transpose(),
            cmap="gray",
            aspect="auto",
        )
        plt.title("neuron(I) spike activity")
        plt.show()

        plt.imshow(
            network["JRecorder", 0]["n.fired", 0, "np"].transpose(),
            cmap="gray",
            aspect="auto",
        )
        plt.title("neuron(J) spike activity")
        plt.show()

    else:
        Network_UI(
            network,
            modules=get_default_UI_modules(["activity"], ["W"]),
            label="my_network_ui",
            group_display_count=1,
        ).show()


# 💁 neuron default start is not from its rest is far up from threshold
# 🙏 make them same as LIF setup of bindsnet
# 💁 TODO: neuron doesn't spike and reset to its reset after it reaches the threshold
# 🙏 Yeah they do, but plot is for trace after spikes so it doesn't show the increases
# 💁 TODO: read gym documentation and bindsnet.libs/environment

if __name__ == "__main__":
    main()
