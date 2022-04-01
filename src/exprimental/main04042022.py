import random
import string

import numpy as np

from PymoNNto import Behaviour, SynapseGroup, Recorder, NeuronGroup, Network
from src.exprimental.core.behaviours.learning import SynapsePairWiseSTDP
from src.exprimental.core.environement.dopamine import DopamineEnvironment
from src.exprimental.core.metrics.metrics import Metrics
from src.exprimental.core.neurons.neurons import LIFNeuron
from src.libs.behaviours import Homeostasis
from src.libs.data_generator_numpy import gen_corpus
from src.libs.helper import (
    behaviour_generator,
    reset_random_seed,
    raster_plots,
)

# from scipy import spatial
# from scipy.spatial.distance import jaccard

# =================   CONFIG    =================

reset_random_seed(42)
language = string.ascii_lowercase + " "
letters = language.strip()
words = ["abc", "omn"]


def spike_stream_i(char):
    spikes = np.zeros(len(letters), dtype=int)
    if char in letters:
        # TODO: this must not be hardcoded, best is bool and 1
        spikes[letters.index(char)] = 90
    return spikes


def get_data(size, prob=0.7):
    corpus = gen_corpus(
        size,
        prob,
        min_length=3,
        max_length=3,
        no_common_chars=False,
        letters_to_use=letters,
        words_to_use=words,
    )
    random.shuffle(corpus)
    sparse_gap = " " * 7
    joined_corpus = sparse_gap.join(corpus) + sparse_gap
    stream_i = [spike_stream_i(char) for char in joined_corpus]
    stream_j = []

    empty_spike = np.empty(len(words))
    empty_spike[:] = np.NaN

    for word in corpus:
        for char in range(len(word) - 1):
            stream_j.append(empty_spike)

        word_spike = np.zeros(len(words), dtype=bool)
        if word in words:
            word_index = words.index(word)
            word_spike[word_index] = 1
        stream_j.append(word_spike)  # spike when see hole word!

        for _ in range(len(sparse_gap)):
            stream_j.append(empty_spike)  # space between words

    assert len(stream_i) == len(stream_j), "stream length mismatch"
    return stream_i, stream_j, corpus


# ================= BEHAVIOURS  =================
class Supervisor(Behaviour):
    __slots__ = ["dopamine_decay", "outputs"]

    def set_variables(self, neurons):
        self.dopamine_decay = 1 - self.get_init_attr("dopamine_decay", 0.0, neurons)
        self.outputs = self.get_init_attr("outputs", [], neurons)

    def new_iteration(self, neurons):
        output = self.outputs[neurons.iteration - 1]
        prediction = neurons.fired

        if np.isnan(output).any():
            DopamineEnvironment.decay(self.dopamine_decay)
            return

        """ Cosine similarity """
        # distance = 1 - spatial.distance.cosine(
        #     re_range_binary(output), re_range_binary(prediction)
        # )
        # DopamineEnvironment.set(distance or -1)  # replace 0.o effect with -1

        """ mismatch similarity """
        distance = [-1.0, 1.0][int((output == prediction).all())]
        DopamineEnvironment.set(distance)

        # DopamineEnvironment.set(-1)
        """ https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jaccard.html """
        # distance = jaccard(output, prediction)
        # DopamineEnvironment.set(-distance or 1.0)


class SynapseDelay(Behaviour):
    __slots__ = [
        "max_delay",
        "delayed_spikes",
        "weight_share",
        "int_delay",
        "delay_mask",
    ]

    def set_variables(self, synapse):
        self.max_delay = self.get_init_attr("max_delay", 0.0, synapse)
        use_shared_weights = self.get_init_attr("use_shared_weights", False, synapse)
        mode = self.get_init_attr("mode", "random", synapse)
        depth_size = 1 if use_shared_weights else synapse.dst.size

        if mode == "random":
            synapse.delay = (
                np.random.random((depth_size, synapse.src.size)) * self.max_delay + 1
            )
        if isinstance(mode, float):
            assert mode != 0, "mode can not be zero"
            synapse.delay = np.ones((depth_size, synapse.src.size)) * mode

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
            """ accumulative shift of weight_share """
            weight_scale[:, :, 0] += synapse.weights_scale[:, :, -1]
        synapse.weights_scale = weight_scale

    def update_delay_float(self, synapse):
        # TODO: synapse.delay = synapse.delay - dw; # {=> in somewhere else}
        synapse.delay = np.clip(np.round(synapse.delay, 1), 0, self.max_delay)
        # print("delay", synapse.delay.flatten())
        """ int_delay: (src.size, dst.size) """
        self.int_delay = np.ceil(synapse.delay).astype(dtype=int)
        """ update delay mask (dst.size, src.size, max_delay) """
        self.delay_mask = np.zeros_like(self.delayed_spikes, dtype=bool)
        for n_idx in range(self.int_delay.shape[0]):
            """ Set neurons in delay index to True """
            for delay, row in zip(self.int_delay[n_idx], self.delay_mask[n_idx]):
                if delay != 0:
                    row[-delay] = True

        # MAYBE MOVE TO ANOTHER FUNCTION MAKE CALL PREDICTABLE
        """ Update weight share based on float delays """
        self.weight_share[:, :, 0] = synapse.delay % 1.0
        weight_share_in_time_t = self.weight_share[:, :, 0]
        # Full weight share for integer delays
        weight_share_in_time_t[weight_share_in_time_t == 0] = 1.0
        self.weight_share[:, :, 0] = weight_share_in_time_t
        self.weight_share[:, :, 1] = 1 - self.weight_share[:, :, 0]


class DerivedLIFNeuron(Behaviour):
    __slots__ = ["stream", "dt"]

    def set_variables(self, n):
        self.dt = self.get_init_attr("dt", 0.1, n)
        n.v_rest = self.get_init_attr("v_rest", -65, n)
        n.v_reset = self.get_init_attr("v_reset", -65, n)
        n.threshold = self.get_init_attr("threshold", -52, n)
        n.v = n.v_rest + n.get_neuron_vec(mode="uniform") * (n.threshold - n.v_reset)
        n.fired = n.get_neuron_vec(mode="zeros") > 0
        n.I = n.get_neuron_vec(mode="zeros")
        n.R = self.get_init_attr("R", 1, n)
        n.tau = self.get_init_attr("tau", 3, n)

    def new_iteration(self, n):
        dv_dt = (n.v_rest - n.v) + n.R * n.I
        n.v += dv_dt * self.dt / n.tau


class Fire(Behaviour):
    def new_iteration(self, n):
        n.fired = n.v >= n.threshold
        if np.sum(n.fired) > 0:
            n.v[n.fired] = n.v_reset


# ================= NETWORK  =================
def main():
    network = Network()
    stream_i_train, stream_j_train, corpus_train = get_data(1000, prob=0.6)
    stream_i_test, stream_j_test, corpus_test = get_data(800, prob=0.6)

    lif_base = {
        "v_rest": -65,
        "v_reset": -65,
        "threshold": -52,
        "dt": 1.0,
        "R": 2,
        "tau": 3,
    }

    letters_ng = NeuronGroup(
        net=network,
        tag="letters",
        size=len(letters),
        behaviour=behaviour_generator(
            [
                LIFNeuron(tag="lif:train", stream=stream_i_train, **lif_base),
                LIFNeuron(tag="lif:test", stream=stream_i_test, **lif_base),
                Recorder(tag="letters-recorder", variables=["n.v", "n.fired"]),
            ]
        ),
    )

    words_ng = NeuronGroup(
        net=network,
        tag="words",
        size=len(words),
        behaviour=behaviour_generator(
            [
                DerivedLIFNeuron(**lif_base),
                Homeostasis(tag="homeostasis", max_ta=-55, min_ta=-70, eta_ip=0.001),
                Fire(),
                #  dopamine_decay should reset a word 1  by at last 3(max delay) time_steps
                Supervisor(
                    tag="supervisor:train", dopamine_decay=1 / 3, outputs=stream_j_train
                ),
                Supervisor(
                    tag="supervisor:test", dopamine_decay=1 / 3, outputs=stream_j_test
                ),
                Metrics(tag="metrics:train", outputs=stream_j_train),
                Metrics(tag="metrics:test", outputs=stream_j_test),
                Recorder(tag="words-recorder", variables=["n.v", "n.fired"]),
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
                SynapseDelay(max_delay=3, mode="random", use_shared_weights=False),
                SynapsePairWiseSTDP(
                    tag="stdp",
                    tau_plus=3.0,
                    tau_minus=3.0,
                    a_plus=6.0,
                    a_minus=6.0,
                    dt=1.0,
                    w_min=0,
                    w_max=4.33,
                    stdp_factor=1.1,
                    delay_epsilon=0.15,
                    weight_decay=0.0,  # 🙅
                ),
            ]
        ),
    )

    network.initialize()
    network.activate_mechanisms(["lif:train", "supervisor:train", "metrics:train"])
    network.deactivate_mechanisms(["lif:test", "supervisor:test", "metrics:test"])
    epochs = 2
    for episode in range(epochs):
        network.iteration = 0
        network.simulate_iterations(len(stream_i_train), measure_block_time=True)
        W = network.SynapseGroups[0].W
        print(
            f"episode={episode} sum={np.sum(W):.1f}, max={np.max(W):.1f}, min={np.min(W):.1f}"
        )
        # raster_plots(network, ngs=["words"])
        network["letters-recorder", 0].reset()
        network["words-recorder", 0].reset()
        network["metrics:train", 0].reset()
        raster_plots(network, ngs=["letters"])
        raster_plots(network, ngs=["words"])

    network.activate_mechanisms(["lif:test", "supervisor:test", "metrics:test"])
    network.deactivate_mechanisms(["lif:train", "supervisor:train", "metrics:train"])
    # Hacky integration, preventing another weight copy!
    network["stdp", 0].recording = False

    network.iteration = 0
    network["words-recorder", 0].clear_cache()
    network["words-recorder", 0].variables = {"n.v": [], "n.fired": []}
    network.simulate_iterations(len(stream_i_test), measure_block_time=True)
    raster_plots(network, ngs=["words"])


if __name__ == "__main__":
    main()
