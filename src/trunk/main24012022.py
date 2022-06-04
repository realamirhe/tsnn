import matplotlib.pyplot as plt
import numpy as np

from PymoNNto import Behaviour, def_dtype, Network, NeuronGroup, Recorder, SynapseGroup
from src.trunk.libs.data_generator_numpy import (
    LETTERS,
    WORDS,
    CORPUS,
    char2spike,
    labels,
)
from src.trunk.libs.helper import reset_random_seed, behaviour_generator

# ================= DATA  =================

input_data = [char2spike(char) for char in CORPUS]
letters_input_data = np.stack(input_data).astype(bool)

"""
- split weight
- dopamine
"""

# ======= DOPAMINE ENVIRONMENT  =============
"""
dopamine is going to act as really global environment variable with effect on all connections not the involved ones.
"""
dopamine = 0


# ================= behaviors  =================
class LIFMechanism(Behaviour):
    def set_variables(self, n):
        self.add_tag("LIFMechanism")
        self.set_init_attrs_as_variables(n)
        n.v = n.get_neuron_vec("uniform") * n.v_rest  # voltage
        n.fired = n.get_neuron_vec() > 0
        n.dt = 0.1

    def new_iteration(self, n):
        n.v += ((n.v_rest - n.v) + n.I) * n.dt

        n.fired = n.v > n.v_threshold
        if np.sum(n.fired) > 0:
            n.v[n.fired] = n.v_reset


class LIFInput(Behaviour):
    def set_variables(self, n):
        for s in n.afferent_synapses["All"]:
            s.W = s.get_synapse_mat("uniform")

        n.I = n.get_neuron_vec()

    def new_iteration(self, n):
        n.I = 90 * n.get_neuron_vec("uniform")

        for s in n.afferent_synapses["GLUTAMATE"]:
            n.I += np.sum(s.W[:, s.src.fired], axis=1)

        for s in n.afferent_synapses["GABA"]:
            n.I -= np.sum(s.W[:, s.src.fired], axis=1)


class ForceSpikeOnLetters(Behaviour):
    """
    If you put the order after LIFInput you must reset fired neuron by yourself
    """

    def new_iteration(self, n):
        n.fired = letters_input_data[n.iteration - 1]


class STDP(Behaviour):
    def set_variables(self, n):
        self.add_tag("STDP")
        n.stdp_factor = self.get_init_attr("stdp_factor", 0.00015, n)
        # TODO: necessitate for stdp on non glutamate synapse type
        self.syn_type = self.get_init_attr("syn_type", "GLUTAMATE", n)
        n.voltage_old = n.get_neuron_vec()

    def new_iteration(self, n):
        global dopamine
        for s in n.afferent_synapses[self.syn_type]:
            pre_post = s.dst.v[:, None] * s.src.voltage_old[None, :]
            stimulus = s.dst.v[:, None] * s.src.v[None, :]
            post_pre = s.dst.voltage_old[:, None] * s.src.v[None, :]
            w_scale = s.w_scale if hasattr(s, "w_scale") else 1
            print(f"{w_scale=}")
            dw = (
                dopamine  # from global environment
                * w_scale  # for delayed connection only
                * n.stdp_factor  # learning stdp factor
                * (pre_post - post_pre + stimulus)  # stdp mechanism
            )

            s.W = np.clip(s.W + dw * s.enabled, 0.0, 10.0)

        n.voltage_old = n.v.copy()


class Homeostasis(Behaviour):
    """
    This mechanism can be used to stabilize the neurons activity.
    https://pymonnto.readthedocs.io/en/latest/Complex_Tutorial/Homeostasis/
    """

    def set_variables(self, n):
        self.add_tag("Homeostatic_Mechanism")

        target_act = self.get_init_attr("target_voltage", 0.05, n)

        self.max_ta = self.get_init_attr("max_ta", target_act, n)
        self.min_ta = self.get_init_attr("min_ta", -target_act, n)

        self.adj_strength = -self.get_init_attr("eta_ip", 0.001, n)

        n.exhaustion = n.get_neuron_vec()

    def new_iteration(self, neurons):
        greater = ((neurons.v > self.max_ta) * -1).astype(def_dtype)
        smaller = ((neurons.v < self.min_ta) * 1).astype(def_dtype)

        greater *= neurons.v - self.max_ta
        smaller *= self.min_ta - neurons.v

        change = (greater + smaller) * self.adj_strength
        neurons.exhaustion += change

        neurons.v -= neurons.exhaustion


class Normalization(Behaviour):
    """
    This module can be used to normalize all synapses connected to a NeuronGroup of a given type.
    https://pymonnto.readthedocs.io/en/latest/Complex_Tutorial/Normalization/
    """

    def set_variables(self, neurons):
        self.add_tag("Normalization")
        self.syn_type = self.get_init_attr("syn_type", "GLUTAMATE", neurons)
        self.norm_factor = self.get_init_attr("norm_factor", 1.0, neurons)
        neurons.temp_weight_sum = neurons.get_neuron_vec()

    def new_iteration(self, neurons):
        neurons.temp_weight_sum *= 0.0

        for s in neurons.afferent_synapses[self.syn_type]:
            s.dst.temp_weight_sum += np.sum(np.abs(s.W), axis=1)

        neurons.temp_weight_sum /= self.norm_factor

        for s in neurons.afferent_synapses[self.syn_type]:
            s.W = s.W / (
                s.dst.temp_weight_sum[:, None] + (s.dst.temp_weight_sum[:, None] == 0)
            )


class Delay(Behaviour):
    """
    This module can be used to delay the input of a NeuronGroup.
    """

    def set_variables(self, n):
        self.add_tag("Delay")
        self.max_delay = self.get_init_attr("max_delay", 0, n)
        self.delay_method = self.get_init_attr("delay_method", "random", n)
        self.delayed_spikes = np.zeros((n.size, self.max_delay + 1), dtype=np.int8)
        n.w_scale = np.ones((n.size, 1), dtype=np.int8)

        if self.delay_method == "zero":
            self.delayed_spikes[:, 0] = 0
        elif self.delay_method == "random":
            self.delayed_spikes[:, 0] = np.random.randint(0, self.max_delay + 1, n.size)
        elif self.delay_method == "constant":
            delay_constant = self.get_init_attr("delay_constant", 0, n)
            # Delay constant must be smaller than max delay
            assert delay_constant <= self.max_delay
            self.delayed_spikes[:, 0] = delay_constant

        self.filling_mask = np.zeros_like(self.delayed_spikes, dtype=bool)
        self.indexing_mask = np.zeros_like(self.delayed_spikes, dtype=bool)
        for neuron_idx in range(n.size):
            delay = self.delayed_spikes[neuron_idx, 0]
            # in a case of zero delay we want all history starts from index 1
            self.filling_mask[neuron_idx, delay or 1 :] = True
            self.indexing_mask[neuron_idx, delay or 1] = True

    def new_iteration(self, n):
        if self.max_delay == 0:
            return

        new_spikes = n.fired.copy()
        # store share wight scale value for the current moment (`or 1` is for non-spiked neurons)
        weight_scale_factor = np.sum(self.delayed_spikes[:, 1:], axis=1, keepdims=True)
        weight_scale_factor[weight_scale_factor == 0] = 1
        for s in n.afferent_synapses["GLUTAMATE"]:  # TODO:CHANGE
            s.w_scale = 1 / weight_scale_factor
        # return value after update is going to be
        n.fired = self.delayed_spikes[:, -1].copy()
        # roll the current status of the delayed_spikes
        self.delayed_spikes[:, 1:] = np.roll(self.delayed_spikes[:, 1:], 1, axis=1)
        self.delayed_spikes[:, 1] = new_spikes

        # copy latest index to rest of delayed_spikes
        self.delayed_spikes = np.where(
            self.filling_mask,
            self.delayed_spikes[self.indexing_mask, None],
            self.delayed_spikes,
        )


class DopamineProvider(Behaviour):
    def set_variables(self, n):
        self.add_tag("Dopamine")
        self.dopamine_scale = self.get_init_attr(
            "dopamine_scale", 0.5, n
        )  # TODO: increase

    def new_iteration(self, n):
        """Evaluate function for dopamine control"""
        global dopamine
        step_label = labels[n.iteration - 1]
        print(f"labels:{step_label} firing:{n.fired}")
        if step_label == -1:
            print("decay dopamine density")
            dopamine *= self.dopamine_scale
        elif n.fired[step_label]:
            print("release dopamine = 1")
            dopamine = 1
        else:
            print("suppress dopamine = -1")
            dopamine = -1


# abc -> abc


# ================= NETWORK  =================
def main():
    network = Network()

    letter_neurons = NeuronGroup(
        net=network,
        tag="letters",
        size=len(LETTERS),
        behaviour=behaviour_generator(
            [
                LIFMechanism(v_rest=-65, v_reset=-80, v_threshold=-10),
                ForceSpikeOnLetters(),
                LIFInput(),
                Delay(max_delay=3, delay_method="constant", delay_constant=3),
                STDP(stdp_factor=0.00015),
                Normalization(norm_factor=10),
                Recorder(tag="letters-recorder", variables=["n.v", "n.fired"]),
            ]
        ),
    )

    word_neurons = NeuronGroup(
        net=network,
        tag="words",
        size=len(WORDS),
        behaviour=behaviour_generator(
            [
                LIFMechanism(v_rest=-65, v_reset=-80, v_threshold=-30),
                LIFInput(),
                STDP(stdp_factor=0.00015),
                Normalization(norm_factor=10),
                DopamineProvider(dopamine_scale=0.001),
                Recorder(tag="words-recorder", variables=["n.v", "n.fired"]),
            ]
        ),
    )

    SynapseGroup(net=network, src=letter_neurons, dst=word_neurons, tag="GLUTAMATE")
    SynapseGroup(net=network, src=word_neurons, dst=word_neurons, tag="GABA")

    network.initialize()
    network.simulate_iterations(letters_input_data.shape[0], measure_block_time=True)

    plt.plot(network["letters-recorder", 0]["n.v", 0])
    plt.title("letters neurons voltage trace")
    plt.xlabel("iterations")
    plt.ylabel("voltage")
    plt.show()

    plt.plot(network["words-recorder", 0]["n.v", 0])
    plt.title("words neurons voltage trace")
    plt.show()

    plt.imshow(
        network["letters-recorder", 0]["n.fired", 0, "np"].transpose(),
        cmap="gray",
        aspect="auto",
    )
    plt.title("letters neurons spike activity")
    plt.show()

    plt.imshow(
        network["words-recorder", 0]["n.fired", 0, "np"].transpose(),
        cmap="gray",
        aspect="auto",
    )
    plt.title("words neurons spike activity")
    plt.show()


if __name__ == "__main__":
    reset_random_seed()
    main()
