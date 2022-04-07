import random
import string

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from PymoNNto import Behaviour, SynapseGroup, Recorder, NeuronGroup, Network
from src.trunk.libs.behaviours import Homeostasis
from src.trunk.libs.data_generator_numpy import gen_corpus
from src.trunk.libs.helper import (
    behaviour_generator,
    reset_random_seed,
    raster_plots,
)

# =================   CONFIG    =================
reset_random_seed(42)
language = string.ascii_lowercase + " "
letters = language.strip()
words = ["abc", "omn"]


def spike_stream_i(char):
    spikes = np.zeros(len(letters), dtype=bool)
    if char in letters:
        spikes[letters.index(char)] = 1
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


# ================= ENVIRONMENT  =================
class DopamineEnvironment:
    dopamine = 0.0

    @classmethod
    def get(cls):
        return cls.dopamine

    @classmethod
    def set(cls, new_dopamine):
        assert -1 <= new_dopamine <= 1
        cls.dopamine = new_dopamine

    @classmethod
    def decay(cls, decay_factor):
        cls.dopamine *= decay_factor


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


class LIFNeuron(Behaviour):
    __slots__ = ["stream", "dt"]

    def set_variables(self, n):
        self.dt = self.get_init_attr("dt", 0.1, n)
        self.stream = self.get_init_attr("stream", None, n)
        n.v_rest = self.get_init_attr("v_rest", -65, n)
        n.v_reset = self.get_init_attr("v_reset", -65, n)
        n.threshold = self.get_init_attr("threshold", -52, n)
        n.v = n.get_neuron_vec(mode="ones") * n.v_rest
        n.fired = n.get_neuron_vec(mode="zeros") > 0
        n.I = n.get_neuron_vec(mode="zeros")
        n.R = self.get_init_attr("R", 1, n)
        n.tau = self.get_init_attr("tau", 3, n)

        # self.set_init_attrs_as_variables(n)
        # n.v = n.get_neuron_vec() * n.v_rest
        # n.spikes = n.get_neuron_vec() > n.threshold
        # n.dt = 1.0

    def new_iteration(self, n):
        n.I = 90 * self.stream[n.iteration - 1]
        dv_dt = (n.v_rest - n.v) + n.R * n.I
        n.v += dv_dt * self.dt / n.tau
        n.fired = n.v >= n.threshold
        if np.sum(n.fired) > 0:
            n.v[n.fired] = n.v_reset


class SynapsePairWiseSTDP(Behaviour):
    __slots__ = [
        "tau_plus",
        "tau_minus",
        "a_plus",
        "a_minus",
        "dt",
        "weight_decay",
        "stdp_factor",
        "delay_epsilon",
        "w_min",
        "w_max",
    ]

    def set_variables(self, synapse):
        synapse.W = synapse.get_synapse_mat("uniform")
        synapse.src.trace = synapse.src.get_neuron_vec()
        synapse.dst.trace = synapse.dst.get_neuron_vec()

        self.tau_plus = self.get_init_attr("tau_plus", 3.0, synapse)
        self.tau_minus = self.get_init_attr("tau_minus", 3.0, synapse)
        self.a_plus = self.get_init_attr("a_plus", 0.1, synapse)
        self.a_minus = self.get_init_attr("a_minus", 0.2, synapse)
        self.dt = self.get_init_attr("dt", 1.0, synapse)
        self.weight_decay = 1 - self.get_init_attr("weight_decay", 0.0, synapse)
        self.stdp_factor = self.get_init_attr("stdp_factor", 1.0, synapse)
        self.delay_epsilon = self.get_init_attr("delay_epsilon", 0.15, synapse)
        self.w_min = self.get_init_attr("w_min", 0.0, synapse)
        self.w_max = self.get_init_attr("w_max", 10.0, synapse)

    def new_iteration(self, synapse):

        if not synapse.recording:
            synapse.dst.I = synapse.W.dot(synapse.src.fired)
            return

        # dx = -synapse.src.trace / self.tau_plus + synapse.src.fired
        # dy = -synapse.dst.trace / self.tau_minus + synapse.dst.fired
        synapse.src.trace += (
            -synapse.src.trace / self.tau_plus + synapse.src.fired
        ) * self.dt
        synapse.dst.trace += (
            -synapse.dst.trace / self.tau_minus + synapse.dst.fired
        ) * self.dt

        dw_minus = (
            -self.a_minus
            * synapse.src.fired[np.newaxis, :]
            * synapse.dst.trace[:, np.newaxis]
        )
        dw_plus = (
            self.a_plus
            * synapse.src.trace[np.newaxis, :]
            * synapse.dst.fired[:, np.newaxis]
        )

        dw = (
            DopamineEnvironment.get()  # from global environment
            * (dw_plus + dw_minus)  # stdp mechanism
            * synapse.weights_scale[:, :, 0]  # weight scale based on the synapse delay
            * self.stdp_factor  # stdp scale factor
            * synapse.enabled  # activation of synapse itself
            * self.dt
        )

        synapse.W = synapse.W * self.weight_decay + dw
        synapse.W = np.clip(synapse.W, self.w_min, self.w_max)

        """ stop condition for delay learning """
        use_shared_delay = dw.shape != synapse.delay.shape
        if use_shared_delay:
            dw = np.mean(dw, axis=0, keepdims=True)

        non_zero_dw = dw != 0
        if non_zero_dw.any():
            should_update = np.min(synapse.delay[non_zero_dw]) > self.delay_epsilon
            if should_update:
                synapse.delay[non_zero_dw] -= dw[non_zero_dw]

        synapse.dst.I = 1e4 * synapse.W.dot(synapse.src.fired)
        # print(np.average(synapse.W))
        # TODO: hardcoded value must be replaced!
        # put clipping mechanism on the neuron itself
        # synapse.dst.I = np.clip(synapse.dst.I, 0, 15)


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


class Metrics(Behaviour):
    __slots__ = ["recording_phase", "outputs", "old_recording", "predictions"]

    def set_variables(self, neurons):
        self.recording_phase = self.get_init_attr("recording_phase", None, neurons)
        self.outputs = self.get_init_attr("outputs", [], neurons)
        self.old_recording = neurons.recording
        self.predictions = []

    def reset(self):
        self.predictions = []

    # recording is different from input
    def new_iteration(self, neurons):
        if (
            self.recording_phase is not None
            and self.recording_phase != neurons.recording
        ):
            return

        if not np.isnan(self.outputs[neurons.iteration - 1]).any():
            # TODO: can append the int here also
            self.predictions.append(neurons.fired)

        if neurons.iteration == len(self.outputs):
            UNK = "unk"
            bit_range = 1 << np.arange(self.outputs[0].size)

            presentation_words = words + [UNK]
            outputs = [o.dot(bit_range) for o in self.outputs if not np.isnan(o).any()]
            predictions = [p.dot(bit_range) for p in self.predictions]

            network_phase = "Testing" if "test" in self.tags[0] else "Training"
            accuracy = accuracy_score(outputs, predictions)

            precision = precision_score(outputs, predictions, average="micro")
            f1 = f1_score(outputs, predictions, average="micro")
            recall = recall_score(outputs, predictions, average="micro")
            # confusion matrix
            cm = confusion_matrix(outputs, predictions)
            cm_sum = cm.sum(axis=1)

            frequencies = np.asarray(np.unique(outputs, return_counts=True)).T
            frequencies_p = np.asarray(np.unique(predictions, return_counts=True)).T

            print(
                "---" * 15,
                f"{network_phase}",
                f"accuracy: {accuracy}",
                f"precision: {precision}",
                f"f1: {f1}",
                f"recall: {recall}",
                f"{','.join(presentation_words)} = {cm.diagonal() / np.where(cm_sum > 0, cm_sum, 1)}",
                "---" * 15,
                f"[Output] frequencies::\n{frequencies}",
                f"[Prediction] frequencies::\n{frequencies_p}",
                sep="\n",
                end="\n\n",
            )

            # display_labels=['none', 'abc', 'omn', 'both']
            cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
            cm_display.plot()
            plt.title(f"{network_phase} Confusion Matrix")
            plt.show()


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
