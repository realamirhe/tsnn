import random
import string

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import jaccard
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from PymoNNto import Behaviour, SynapseGroup, Recorder, NeuronGroup, Network
from src.trunk.libs.data_generator_numpy import gen_corpus
from src.trunk.libs.helper import (
    behaviour_generator,
    reset_random_seed,
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
    joined_corpus = " ".join(corpus) + " "
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
        # print("dopamine => ", "increase" if cls.dopamine < new_dopamine else "decrease")
        cls.dopamine = new_dopamine

    @classmethod
    def decay(cls, decay_factor):
        # print(
        #     "decay dopamine 🔻",
        #     decay_factor,
        #     f"from {cls.dopamine} => {cls.dopamine * decay_factor}",
        # )
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
        # distance = [-1.0, 1.0][int((output == prediction).all())]
        # DopamineEnvironment.set(distance)

        # DopamineEnvironment.set(-1)
        """ https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jaccard.html """
        distance = jaccard(output, prediction)
        DopamineEnvironment.set(-distance or 1.0)


class LIFNeuron(Behaviour):
    __slots__ = ["stream", "dt"]

    def set_variables(self, neurons):
        neurons.v_rest = self.get_init_attr("v_rest", -65, neurons)
        neurons.v_reset = self.get_init_attr("v_reset", -65, neurons)
        neurons.v_threshold = self.get_init_attr("v_threshold", -32, neurons)
        self.dt = self.get_init_attr("dt", 1.0, neurons)
        self.stream = self.get_init_attr("stream", None, neurons)

        # TODO: default voltage
        neurons.v = neurons.v_rest + neurons.get_neuron_vec("uniform") * 10
        neurons.fired = neurons.get_neuron_vec("zeros") > 0
        neurons.I = neurons.get_neuron_vec(mode="zeros")

    def new_iteration(self, n):
        n.v += ((n.v_rest - n.v) + n.I) * self.dt
        if self.stream is not None:
            # todo: if problem in j stream replace [:] with something else
            n.fired[:] = self.stream[n.iteration - 1]
        else:
            n.fired = n.v > n.v_threshold

        if np.sum(n.fired) > 0:
            n.v[n.fired] = n.v_reset

        n.I = 90 * n.get_neuron_vec("uniform")
        for s in n.afferent_synapses.get("GLUTAMATE", []):
            n.I += np.sum(s.W[:, s.src.fired], axis=1)

        for s in n.afferent_synapses.get("GABA", []):
            n.I -= np.sum(s.W[:, s.src.fired], axis=1)


class SynapseSTDP(Behaviour):
    __slots__ = ["weight_decay", "stdp_factor", "delay_epsilon", "w_min", "w_max"]

    def set_variables(self, synapse):
        synapse.W = synapse.get_synapse_mat("uniform")

        self.weight_decay = 1 - self.get_init_attr("weight_decay", 0.0, synapse)
        self.stdp_factor = self.get_init_attr("stdp_factor", 1.0, synapse)
        self.delay_epsilon = self.get_init_attr("delay_epsilon", 0.15, synapse)

        self.w_min = self.get_init_attr("w_min", 0.0, synapse)
        self.w_max = self.get_init_attr("w_max", 10.0, synapse)

        synapse.src.voltage_old = synapse.src.get_neuron_vec(mode="zeros")
        synapse.dst.voltage_old = synapse.dst.get_neuron_vec(mode="zeros")

    def new_iteration(self, synapse):
        if not synapse.recording:
            return

        pre_post = synapse.dst.v[:, np.newaxis] * synapse.src.voltage_old[np.newaxis, :]
        stimulus = synapse.dst.v[:, np.newaxis] * synapse.src.v[np.newaxis, :]
        post_pre = synapse.dst.voltage_old[:, np.newaxis] * synapse.src.v[np.newaxis, :]

        dw = (
            DopamineEnvironment.get()  # from global environment
            * (pre_post - post_pre + stimulus)  # stdp mechanism
            * synapse.weights_scale[:, :, 0]  # weight scale based on the synapse delay
            * self.stdp_factor  # stdp scale factor
            * synapse.enabled  # activation of synapse itself
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

        # This will differ from the original stdp mechanism and must be added to the latest synapse
        synapse.src.voltage_old = synapse.src.v.copy()
        synapse.dst.voltage_old = synapse.dst.v.copy()


class SynapseSTDPET(Behaviour):
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
    """
    Let x and y be the spike trace variables of pre- and post-synaptic neurons.
    These trace variables are modified through time with:
    dx/dt = -x/tau_plus + pre.spikes,
    dy/dt = -y/tau_minus + post.spikes,
    where tau_plus and tau_minus define the time window of STDP. Then, the synaptic
    weights can are updated with the given equation:
    dw/dt = a_plus * x * post.spikes - a_minus * y * pre.spikes,
    where a_plus and a_minus define the intensity of weight change.
    We assume that synapses have dt, w_min, and w_max and neurons have spikes.
    Args:
        tau_plus (float): pre-post time window.
        tau_minus (float): post-pre time window.
        a_plus (float): pre-post intensity.
        a_minus (float): post-pre intensity.
        dt: (flaot): time step.
    """

    def set_variables(self, synapse):
        synapse.src.v = synapse.src.get_neuron_vec()
        synapse.dst.v = synapse.dst.get_neuron_vec()
        synapse.W = synapse.get_synapse_mat("uniform")

        self.tau_plus = self.get_init_attr("tau_plus", 3.0, synapse)
        self.tau_minus = self.get_init_attr("tau_minus", 3.0, synapse)
        self.a_plus = self.get_init_attr("a_plus", 0.01, synapse)
        self.a_minus = self.get_init_attr("a_minus", 2.0, synapse)
        self.dt = self.get_init_attr("dt", 1.0, synapse)
        self.weight_decay = 1 - self.get_init_attr("weight_decay", 0.0, synapse)
        self.stdp_factor = self.get_init_attr("stdp_factor", 1.0, synapse)
        self.delay_epsilon = self.get_init_attr("delay_epsilon", 0.15, synapse)
        self.w_min = self.get_init_attr("w_min", 0.0, synapse)
        self.w_max = self.get_init_attr("w_max", 10.0, synapse)

    def new_iteration(self, synapse):
        """
        Single step of STDP.
        Args:
            synapse (SynapseGroup): the synapse to which STDP is applied.
        """
        dx = -synapse.src.v / self.tau_plus + synapse.src.fired
        dy = -synapse.dst.v / self.tau_minus + synapse.dst.fired
        synapse.src.v += dx * self.dt
        synapse.dst.v += dy * self.dt

        dw_minus = (
            -self.a_minus
            * synapse.src.fired[np.newaxis, :]
            * synapse.dst.v[:, np.newaxis]
        )
        dw_plus = (
            self.a_plus
            * synapse.src.v[np.newaxis, :]
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

        del dx
        del dy
        del dw_minus
        del dw_plus
        del dw


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
        depth_size = 1 if use_shared_weights else synapse.dst.size

        synapse.delay = (
            np.random.random((depth_size, synapse.src.size)) * self.max_delay + 1
        )
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


class SynapseSTDPWithOutDelay(Behaviour):
    __slots__ = ["weight_decay", "stdp_factor", "w_min", "w_max"]

    def set_variables(self, synapse):
        self.add_tag("STDPWithOutDelay")
        synapse.W = synapse.get_synapse_mat("uniform")

        self.weight_decay = 1 - self.get_init_attr("weight_decay", 0.0, synapse)
        self.stdp_factor = self.get_init_attr("stdp_factor", 1.0, synapse)

        self.w_min = self.get_init_attr("w_min", 0.0, synapse)
        self.w_max = self.get_init_attr("w_max", 10.0, synapse)

        synapse.src.voltage_old = synapse.src.get_neuron_vec(mode="zeros")
        synapse.dst.voltage_old = synapse.dst.get_neuron_vec(mode="zeros")

    def new_iteration(self, synapse):
        if not synapse.recording:
            return

        pre_post = synapse.dst.v[:, np.newaxis] * synapse.src.voltage_old[np.newaxis, :]

        stimulus = synapse.dst.v[:, np.newaxis] * synapse.src.v[np.newaxis, :]
        post_pre = synapse.dst.voltage_old[:, np.newaxis] * synapse.src.v[np.newaxis, :]

        dw = (
            DopamineEnvironment.get()  # from global environment
            * (pre_post - post_pre + stimulus)  # stdp mechanism
            * self.stdp_factor  # stdp scale factor
            * synapse.enabled  # activation of synapse itself (todo)!!
        )
        synapse.W = synapse.W * self.weight_decay + dw
        synapse.W = np.clip(synapse.W, self.w_min, self.w_max)

        # This will differ from the original stdp mechanism and must be added to the latest synapse
        synapse.src.voltage_old = synapse.src.v.copy()
        synapse.dst.voltage_old = synapse.dst.v.copy()


class Metrics(Behaviour):
    __slots__ = ["recording_phase", "outputs", "old_recording", "predictions"]

    def set_variables(self, neurons):
        self.recording_phase = self.get_init_attr("recording_phase", None, neurons)
        self.outputs = self.get_init_attr("outputs", [], neurons)
        self.old_recording = neurons.recording
        self.predictions = []

    def reset(self):
        self.predictions = []

    def new_iteration(self, neurons):
        # recording is different from input
        if (
            self.recording_phase is not None
            and self.recording_phase != neurons.recording
        ):
            return

        if not np.isnan(self.outputs[neurons.iteration - 1]).any():
            self.predictions.append(neurons.fired)

        if neurons.iteration == len(self.outputs):
            UNK = "unk"
            bit_range = 1 << np.arange(self.outputs[0].size)

            presentation_words = words + [UNK]
            outputs = [o.dot(bit_range) for o in self.outputs if not np.isnan(o).any()]
            predictions = [p.dot(bit_range) for p in self.predictions]

            network_phase = "Training" if neurons.recording else "Testing"
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

            cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
            cm_display.plot()
            plt.title(f"{network_phase} Confusion Matrix")
            plt.show()


# ================= NETWORK  =================
def main():
    network = Network()
    stream_i_train, stream_j_train, corpus_train = get_data(1000, prob=0.6)
    stream_i_test, stream_j_test, corpus_test = get_data(800, prob=0.6)

    letters_ng = NeuronGroup(
        net=network,
        tag="letters",
        size=len(letters),
        behaviour=behaviour_generator(
            [
                LIFNeuron(
                    tag="lif:train",
                    v_rest=-65,
                    v_reset=-65,
                    v_threshold=-12,
                    stream=stream_i_train,
                ),
                LIFNeuron(
                    tag="lif:test",
                    v_rest=-65,
                    v_reset=-65,
                    v_threshold=-12,
                    stream=stream_i_test,
                ),
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
                LIFNeuron(v_rest=-65, v_reset=-65, v_threshold=-52),
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
                SynapseDelay(max_delay=3, use_shared_weights=False),
                # SynapseSTDP(
                #     tag="stdp",
                #     weight_decay=0.1,
                #     stdp_factor=0.0015,
                #     delay_epsilon=0.15,
                #     w_min=-10.0,
                #     w_max=10.0,
                # ),
                SynapseSTDPET(
                    tag="stdp",
                    tau_plus=3.0,
                    tau_minus=3.0,
                    a_plus=20.0,
                    a_minus=30.0,
                    dt=1.0,
                    w_min=-300.0,
                    w_max=20.0,
                    weight_decay=0,  # 🙅
                ),
            ]
        ),
    )

    network.initialize()
    network.activate_mechanisms(["lif:train", "supervisor:train", "metrics:train"])
    network.deactivate_mechanisms(["lif:test", "supervisor:test", "metrics:test"])
    epochs = 1
    for episode in range(epochs):
        network.iteration = 0
        network.simulate_iterations(len(stream_i_train), measure_block_time=True)
        W = network.SynapseGroups[0].W
        print(
            f"episode={episode} sum={np.sum(W):.1f}, max={np.max(W):.1f}, min={np.min(W):.1f}"
        )

        network["letters-recorder", 0].reset()
        network["words-recorder", 0].reset()
        network["metrics:train", 0].reset()

    network.activate_mechanisms(["lif:test", "supervisor:test", "metrics:test"])
    network.deactivate_mechanisms(
        ["lif:train", "supervisor:train", "metrics:train", "stdp"]
    )
    network.iteration = 0
    network.simulate_iterations(len(stream_i_test), measure_block_time=True)


if __name__ == "__main__":
    main()
