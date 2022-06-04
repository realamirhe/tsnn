import itertools
import random

import numpy as np

from PymoNNto import Behaviour, SynapseGroup, Recorder, NeuronGroup, Network

from src.trunk.libs.helper import (
    behaviour_generator,
    voltage_visualizer,
    spike_visualizer,
    reset_random_seed,
)

# =================   CONFIG    =================
reset_random_seed(42)
language = "abc "


def spike_stream_j(characters):
    spikes = np.zeros(1, dtype=bool)
    spikes[0] = characters
    return spikes


def spike_stream_i(characters):
    spikes = np.zeros(len(language), dtype=bool)
    for char in characters:
        spikes[language.index(char)] = 1
    return spikes


letters = language.strip()
streams = (
    ["".join(permutation) for permutation in itertools.permutations(letters, 1)]
    + ["".join(permutation) for permutation in itertools.permutations(letters, 2)]
    + ["".join(permutation) for permutation in itertools.permutations(letters, 3)]
    + ["".join(permutation) for permutation in itertools.permutations(letters, 4)]
)
streams += ["abc"] * 50  # true neurons
streams = [(characters, characters == "abc") for characters in streams]
random.shuffle(streams)
character_streams_i = []
character_streams_j = []
for (letters_neurons, words_neurons) in streams:
    character_spikes = [[False]] * len(letters_neurons)
    if words_neurons:
        character_spikes[-1] = [True]
    character_streams_j.extend(character_spikes)
    character_streams_j.append([False])  # for space
    character_streams_i.append(letters_neurons)

character_streams_i = " ".join(character_streams_i)

stream_j = character_streams_j
stream_i = [spike_stream_i(i) for i in character_streams_i]
print("stream j", stream_j)
print("stream i", stream_i)
ITERATIONS = len(stream_i)


# ================= ENVIRONMENT  =================
class DopamineEnvironment:
    dopamine = 0.0

    @classmethod
    def get(cls):
        return cls.dopamine

    @classmethod
    def set(cls, new_dopamine):
        assert -1 <= new_dopamine <= 1
        print("dopamine => ", "increase" if cls.dopamine < new_dopamine else "decrease")
        cls.dopamine = new_dopamine

    @classmethod
    def decay(cls, decay_factor):
        print(
            "decay dopamine 🔻",
            decay_factor,
            f"from {cls.dopamine} => {cls.dopamine * decay_factor}",
        )
        cls.dopamine *= decay_factor


# ================= BEHAVIOURS  =================
class Supervisor(Behaviour):
    """
    objective reached => release
    objective missed => punish
    o.w. => decay
    📒 TODO: decay for dopamine must be set wisely to be reduced in n steps!
    📒 TODO: big question rises here, make sure it is scalable and general!!
    """

    def set_variables(self, neurons):
        self.dopamine_decay = 1 - self.get_init_attr("dopamine_decay", 0.0, neurons)
        self.outputs = self.get_init_attr("outputs", [], neurons)

    def new_iteration(self, neurons):
        output = self.outputs[neurons.iteration - 1]
        prediction = neurons.fired
        print(f"iteration={neurons.iteration} {output=} {prediction=}")
        if prediction[0]:
            if output[0]:
                DopamineEnvironment.decay(self.dopamine_decay + 0.01)
            else:  # must decrease w
                DopamineEnvironment.set(-1)
        else:
            if output[0]:  # must increase w
                DopamineEnvironment.set(1.0)
            else:
                DopamineEnvironment.decay(self.dopamine_decay)


class LIFNeuron(Behaviour):
    def set_variables(self, neurons):
        self.add_tag("LIFNeuron")
        self.set_init_attrs_as_variables(neurons)
        # TODO: default voltage
        neurons.v = neurons.v_rest + neurons.get_neuron_vec("uniform") * 10
        neurons.fired = neurons.get_neuron_vec("zeros") > 0
        neurons.dt = getattr(neurons, "dt", 0.1)  # TODO: default dt
        neurons.I = neurons.get_neuron_vec(mode="zeros")
        self.stream = self.get_init_attr("stream", None, neurons)

    def new_iteration(self, n):
        n.v += ((n.v_rest - n.v) + n.I) * n.dt
        if self.stream is not None:
            # todo: if problem in j stream replace [:] with something else
            n.fired[:] = self.stream[n.iteration - 1]
        else:
            n.fired = n.v > n.v_threshold

        if np.sum(n.fired) > 0:
            n.v[n.fired] = n.v_reset

        n.I = 90 * n.get_neuron_vec("uniform")
        for s in getattr(n.afferent_synapses, "GLUTAMATE", []):
            n.I += np.sum(s.W[:, s.src.fired], axis=1)

        for s in getattr(n.afferent_synapses, "GABA", []):
            n.I -= np.sum(s.W[:, s.src.fired], axis=1)


class SynapseSTDP(Behaviour):
    def set_variables(self, synapse):
        self.add_tag("STDP")
        synapse.W = synapse.get_synapse_mat("uniform")

        self.weight_decay = 1 - self.get_init_attr("weight_decay", 0.0, synapse)
        self.stdp_factor = self.get_init_attr("stdp_factor", 1.0, synapse)
        self.delay_epsilon = self.get_init_attr("delay_epsilon", 0.15, synapse)

        self.w_min = self.get_init_attr("w_min", 0.0, synapse)
        self.w_max = self.get_init_attr("w_max", 10.0, synapse)

        synapse.src.voltage_old = synapse.src.get_neuron_vec(mode="zeros")
        synapse.dst.voltage_old = synapse.dst.get_neuron_vec(mode="zeros")

    def new_iteration(self, synapse):
        pre_post = synapse.dst.v[:, np.newaxis] * synapse.src.voltage_old[np.newaxis, :]

        stimulus = synapse.dst.v[:, np.newaxis] * synapse.src.v[np.newaxis, :]
        post_pre = synapse.dst.voltage_old[:, np.newaxis] * synapse.src.v[np.newaxis, :]

        dw = (
            DopamineEnvironment.get()  # from global environment
            * (pre_post - post_pre + stimulus)  # stdp mechanism
            * synapse.weights_scale[:, :, 0]  # weight scale based on the synapse delay
            * self.stdp_factor  # stdp scale factor
            * synapse.enabled  # activation of synapse itself (todo)!!
        )
        synapse.W = synapse.W * self.weight_decay + dw
        synapse.W = np.clip(synapse.W, self.w_min, self.w_max)

        """ stop condition for delay learning """
        update_delay_mask = np.min(synapse.delay, axis=1) > self.delay_epsilon
        synapse.delay[update_delay_mask] -= dw[update_delay_mask]
        # This will differ from the original stdp mechanism and must be added to the latest synapse
        synapse.src.voltage_old = synapse.src.v.copy()
        synapse.dst.voltage_old = synapse.dst.v.copy()


class SynapseDelay(Behaviour):
    def set_variables(self, synapse):
        self.max_delay = self.get_init_attr("max_delay", 0.0, synapse)
        synapse.delay = synapse.get_synapse_mat("uniform") * self.max_delay
        self.delayed_spikes = np.zeros(
            (synapse.dst.size, synapse.src.size, self.max_delay), dtype=bool
        )
        self.weight_share = np.ones(
            (synapse.dst.size, synapse.src.size, 2), dtype=np.float32
        )
        self.weight_share[:, :, -1] = 0.0

        self.update_delay_float(synapse)

    def new_iteration(self, synapse):
        # TBD: check if this is correct
        # if self.max_delay == 0:
        #     return

        self.update_delay_float(synapse)

        new_spikes = synapse.src.fired.copy()

        """ TBD: neurons activity is based on one of its own delayed activity """
        """ Spike immediately for neurons with zero delay """
        t_spikes = self.delayed_spikes[:, :, -1]
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

        weight_scale = t_spikes[:, :, np.newaxis] * self.weight_share
        if hasattr(synapse, "weight_scale"):
            """accumulative shift of weight_share"""
            weight_scale[:, :, 0] += synapse.weights_scale[:, :, -1]
        synapse.weights_scale = weight_scale

    def update_delay_float(self, synapse):
        # TODO: synapse.delay = synapse.delay - dw; # {=> in somewhere else}
        synapse.delay = np.clip(np.round(synapse.delay, 1), 0, self.max_delay)
        """ int_delay: (src.size, dst.size) """
        self.int_delay = np.ceil(synapse.delay).astype(dtype=int)
        """ update delay mask (dst.size, src.size, max_delay) """
        self.delay_mask = np.zeros_like(self.delayed_spikes, dtype=bool)
        for n_idx in range(self.int_delay.shape[0]):
            """Set neurons in delay index to True"""
            for delay, row in zip(self.int_delay[n_idx], self.delay_mask[n_idx]):
                if delay != 0:
                    row[-delay] = True

        # MAYBE MOVE TO ANOTHER FUNCTION MAKE CALL PREDICTABLE
        """ Update weight share based on float delays """
        self.weight_share[:, :, 0] = synapse.delay % 1.0
        weight_share_in_time_t = self.weight_share[:, :, 0]
        weight_share_in_time_t[weight_share_in_time_t == 0] = 1.0
        self.weight_share[:, :, 0] = weight_share_in_time_t
        self.weight_share[:, :, 1] = 1 - self.weight_share[:, :, 0]


# ================= NETWORK  =================
def main():
    network = Network()
    letters_ng = NeuronGroup(
        net=network,
        tag="letters",
        size=len(language),
        behaviour=behaviour_generator(
            [
                LIFNeuron(v_rest=-65, v_reset=-65, v_threshold=-52, stream=stream_i),
                Recorder(tag="letters-recorder", variables=["n.v", "n.fired"]),
            ]
        ),
    )

    words_ng = NeuronGroup(
        net=network,
        tag="words",
        size=1,
        behaviour=behaviour_generator(
            [
                LIFNeuron(v_rest=-65, v_reset=-65, v_threshold=-52),
                Supervisor(dopamine_decay=0.1, outputs=stream_j),
                Recorder(tag="word-recorder", variables=["n.v", "n.fired"]),
            ]
        ),
    )

    SynapseGroup(
        net=network,
        src=letters_ng,
        dst=words_ng,
        tag="GLUTAMATE",
        behaviour=behaviour_generator(
            [
                SynapseDelay(max_delay=3),
                SynapseSTDP(weight_decay=0.1, stdp_factor=0.00015, delay_epsilon=0.15),
            ]
        ),
    )
    network.initialize()
    print("start")
    print(network.SynapseGroups[0].W)
    network.simulate_iterations(ITERATIONS, measure_block_time=True)
    print("finished")
    print(network.SynapseGroups[0].W)

    voltage_visualizer(
        network["letters-recorder", 0]["n.v", 0], title="letters.voltage (trace)"
    )
    voltage_visualizer(
        network["word-recorder", 0]["n.v", 0], title="words.voltage (trace)"
    )
    spike_visualizer(
        network["letters-recorder", 0]["n.fired", 0, "np"].transpose(),
        title="letters spike activity",
    )
    spike_visualizer(
        network["word-recorder", 0]["n.fired", 0, "np"].transpose(),
        title="words spike activity",
    )


if __name__ == "__main__":
    main()
