import numpy as np

from PymoNNto import Behaviour, SynapseGroup, Recorder, NeuronGroup, Network

# from PymoNNto.Exploration.Network_UI import get_default_UI_modules, Network_UI
# from src.libs import behaviours
# from src.libs.data_generator_numpy import stream_generator_for_character
# from src.libs.environment import set_dopamine, get_dopamine
from src.trunk.libs.helper import (
    behaviour_generator,
    voltage_visualizer,
    spike_visualizer,
    reset_random_seed,
)

# =================   CONFIG    =================
reset_random_seed(42)


def spike_stream_j(characters):
    spikes = np.zeros(1, dtype=bool)
    spikes[0] = characters
    return spikes


def spike_stream_i(characters):
    spikes = np.zeros(len(language), dtype=bool)
    for char in characters:
        spikes[language.index(char)] = 1
    return spikes


language = "abcs"
# increase weights
# streams = [("a", 0), ("abc", 1), ("bc", 0), ("abc", 1), ("abc", 1), ("b", 0)]
# decrease weights
streams = [("a", 0), ("a", 0), ("abc", 0), ("bc", 1), ("ac", 1), ("abc", 0)]

# ("as", False),
# ("ac", False),
# ("abc", False),
# ("sb", False),
# ("b", False),
# ("ba", False),
# ("ca", False),
# ("abc", False),
# ("s", False),
# ("s", False),
# ("as", False),
# ("s", False),
# ("sa", False),
# ("sa", False),
# ("ss", False),
# ("sa", False),
# ("abcs", False),
# ("abcs", False),
# ("abcs", False),
# ("abcs", False),

stream_j = [spike_stream_j(j) for _, j in streams]
stream_i = [spike_stream_i(i) for i, _ in streams]
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
        print(stream_i[neurons.iteration - 1], output, prediction)

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

        # if (prediction == output).all():
        #     if np.sum(prediction) > 0:
        #         print("📒 release dopamine")
        #         DopamineEnvironment.set(1.0)
        #     else:
        #         print("📒 dopamine decay")
        #         DopamineEnvironment.decay(self.dopamine_decay)
        # else:
        #     print("📒 punished prediction because of unwanted spikes")
        #     DopamineEnvironment.set(-1.0)

        # spike = punishment
        # if np.sum(neurons.fired[output]) == np.sum(output):
        #     DopamineEnvironment.set([1.0, -1.0][prediction])
        # elif output == 1:  # spike = reward
        #     DopamineEnvironment.set([-1.0, 1.0][prediction])
        # else:
        #     DopamineEnvironment.decay(self.dopamine_decay)


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


class SynapseWeight(Behaviour):
    def set_variables(self, synapse):
        synapse.W = synapse.get_synapse_mat("uniform")


class SynapseSTDP(Behaviour):
    def set_variables(self, synapse):
        self.add_tag("STDP")
        self.weight_decay = 1 - self.get_init_attr("weight_decay", 0.0, synapse)
        self.stdp_factor = self.get_init_attr("stdp_factor", 1.0, synapse)

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
            * self.stdp_factor  # stdp scale factor
            * synapse.enabled  # activation of synapse itself (todo)!!
        )
        print("dw => ", dw)
        synapse.W = synapse.W * self.weight_decay + dw
        synapse.W = np.clip(synapse.W, self.w_min, self.w_max)
        # This will differ from the original stdp mechanism and must be added to the latest synapse
        synapse.src.voltage_old = synapse.src.v.copy()
        synapse.dst.voltage_old = synapse.dst.v.copy()


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
                # STDP(stdp_factor=0.015),
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
            [SynapseWeight(), SynapseSTDP(weight_decay=0.1, stdp_factor=0.00015)]
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
