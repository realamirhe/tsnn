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
    if size < 10:
        print(corpus)
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
    return stream_i, stream_j


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
    def set_variables(self, neurons):
        self.add_tag("Supervisor")
        self.dopamine_decay = 1 - self.get_init_attr("dopamine_decay", 0.0, neurons)
        self.outputs = self.get_init_attr("outputs", [], neurons)

    def new_iteration(self, neurons):
        output = self.outputs[neurons.iteration - 1]
        prediction = neurons.fired

        if np.isnan(output).any():
            DopamineEnvironment.decay(self.dopamine_decay)
            return

        """ Cosine similarity """
        # DopamineEnvironment.set(
        #     (
        #         1
        #         - spatial.distance.cosine(
        #             re_range_binary(output), re_range_binary(prediction)
        #         )
        #     )
        #     or -1
        # ) # replace 0.o effect with -1

        """ mismatch similarity """
        # if (output == prediction).all():
        #     DopamineEnvironment.set(1.0)
        # else:
        #     DopamineEnvironment.set(-1.0)

        """ https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jaccard.html """
        distance = jaccard(output, prediction)
        DopamineEnvironment.set(-distance or 1.0)


class LIFNeuron(Behaviour):
    def set_variables(self, neurons):
        self.add_tag("LIFNeuron")
        neurons.v_rest = self.get_init_attr("v_rest", -65, neurons)
        neurons.v_reset = self.get_init_attr("v_reset", -65, neurons)
        neurons.v_threshold = self.get_init_attr("v_threshold", -32, neurons)
        neurons.dt = self.get_init_attr("dt", 1.0, neurons)
        self.stream = self.get_init_attr("stream", None, neurons)

        # TODO: default voltage
        neurons.v = neurons.v_rest + neurons.get_neuron_vec("uniform") * 10
        neurons.fired = neurons.get_neuron_vec("zeros") > 0
        neurons.I = neurons.get_neuron_vec(mode="zeros")

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
        for s in n.afferent_synapses.get("GLUTAMATE", []):
            n.I += np.sum(s.W[:, s.src.fired], axis=1)

        for s in n.afferent_synapses.get("GABA", []):
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
            * synapse.enabled  # activation of synapse itself (todo)!!
        )
        # print("dw => ", dw)
        synapse.W = synapse.W * self.weight_decay + dw
        synapse.W = np.clip(synapse.W, self.w_min, self.w_max)

        """ stop condition for delay learning """
        update_delay_mask = np.min(synapse.delay, axis=1) > self.delay_epsilon

        # Update whole delay matrix when the delay is shared between the states
        if update_delay_mask.size == 1:
            if update_delay_mask[0]:
                synapse.delay -= np.mean(dw, axis=0, keepdims=True)
        else:
            synapse.delay[update_delay_mask] -= dw[update_delay_mask]

        # This will differ from the original stdp mechanism and must be added to the latest synapse
        synapse.src.voltage_old = synapse.src.v.copy()
        synapse.dst.voltage_old = synapse.dst.v.copy()


class STDPET(Behaviour):
    def set_variables(self, synapses):
        self.add_tag("STDP")
        self.set_init_attrs_as_variables(synapses)
        synapses.src.trace = synapses.src.get_neuron_vec()
        synapses.dst.trace = synapses.dst.get_neuron_vec()

    def new_iteration(self, synapses):
        dx = -synapses.src.trace / synapses.tau_plus + synapses.src.fired
        dy = -synapses.dst.trace / synapses.tau_minus + synapses.dst.fired
        synapses.src.trace += dx * synapses.src.dt
        synapses.dst.trace += dy * synapses.dst.dt
        dw_minus = -synapses.a_minus * synapses.dst.trace * synapses.src.fired
        dw_plus = synapses.a_plus * synapses.src.trace * synapses.dst.fired
        synapses.W = (
            np.clip(
                synapses.W + (dw_plus + dw_minus) * synapses.src.dt,
                synapses.w_min,
                synapses.w_max,
            )
            * synapses.mask
        )


class SynapseDelay(Behaviour):
    def set_variables(self, synapse):
        self.add_tag("Delay")
        self.max_delay = self.get_init_attr("max_delay", 0.0, synapse)
        use_shared_weights = self.get_init_attr("use_shared_weights", False, synapse)
        depth_size = 1 if use_shared_weights else synapse.dst.size

        synapse.delay = (
            np.random.random((depth_size, synapse.src.size)) * self.max_delay
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
    def set_variables(self, neurons):
        self.add_tag("Metrics")

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

        # [1, 1] -> 0
        # OUTPUT
        # [1, 0] -> 0 :abc:
        # [0, 0] -> 0 :xyw:
        # [0, 1] -> 1 :omn:

        # // -----------
        # TODO:
        # [0, 1] -> 0
        # [1, 0] -> 0 :abc:
        # [0, 0] -> 0 :xyw:
        # [0, 1] -> 1 :omn:
        UNK = len(words)
        presentation_words = words + ["UNK"]
        if neurons.iteration == len(self.outputs):
            outputs = [str(o) for o in self.outputs if not np.isnan(o).any()]
            predictions = [str(p) for p in self.predictions]

            network_phase = "Training" if neurons.recording else "Testing"
            accuracy = accuracy_score(outputs, predictions)
            precision = precision_score(outputs, predictions, average="micro")
            f1 = f1_score(outputs, predictions, average="micro")
            recall = recall_score(outputs, predictions, average="micro")
            # confusion matrix
            cm = confusion_matrix(outputs, predictions)
            cm_sum = cm.sum(axis=1)

            (unique, counts) = np.unique(outputs, return_counts=True)
            # frequencies = np.asarray((unique, counts), dtype=object).T
            # frequencies[:, 0] = np.array(presentation_words)[
            #     frequencies[:, 0].astype(int)
            # ]

            print(
                "---" * 15,
                f"{network_phase}",
                f"accuracy: {accuracy}",
                f"precision: {precision}",
                f"f1: {f1}",
                f"recall: {recall}",
                f"{','.join(presentation_words)} = {cm.diagonal() / np.where(cm_sum > 0, cm_sum, 1)}",
                "---" * 15,
                # f"frequencies {frequencies}",
                sep="\n",
                end="\n\n",
            )

            cm_display = ConfusionMatrixDisplay(
                # confusion_matrix=cm, display_labels=presentation_words
                confusion_matrix=cm,
                # display_labels=np.unique(predictions),
            )
            cm_display.plot()
            plt.title(f"{network_phase} Confusion Matrix")
            plt.show()


# [1, 0] [0, 1], [0, 0], [1,1]

# ================= NETWORK  =================
def main():
    network = Network()
    stream_i, stream_j = get_data(1000, prob=0.85)
    letters_ng = NeuronGroup(
        net=network,
        tag="letters",
        size=len(letters),
        behaviour=behaviour_generator(
            [
                LIFNeuron(v_rest=-65, v_reset=-65, v_threshold=-12, stream=stream_i),
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
                Supervisor(dopamine_decay=0.1, outputs=stream_j),
                Metrics(outputs=stream_j),
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
                SynapseSTDP(
                    weight_decay=0.1,
                    stdp_factor=0.0015,
                    delay_epsilon=0.15,
                    w_min=0,
                    w_max=10.0,
                ),
            ]
        ),
    )

    network.initialize()
    print("start")
    print(np.sum(network.SynapseGroups[0].W))
    network.simulate_iterations(len(stream_i), measure_block_time=False)
    print("finished")
    print(np.sum(network.SynapseGroups[0].W))

    # voltage_plots(network, ngs=["letters", "words"])
    raster_plots(network, ngs=["letters", "words"])

    # Testing purposes
    stream_i, stream_j = get_data(800, prob=0.6)
    network.recording_off()
    network.iteration = 0

    network["letters", 0].behaviour[1].stream = stream_i  # LIFNeuron

    network["letters", 0].recording = True
    network["letters", 0]["letters-recorder", 0].clear_cache()
    network["letters", 0]["letters-recorder", 0].variables = {"n.v": [], "n.fired": []}

    network["words", 0].behaviour[2].outputs = stream_j  # supervisor
    network["words", 0].behaviour[3].outputs = stream_j  # metrics
    network["words", 0].behaviour[3].reset()  # metrics

    network["words", 0].recording = True
    network["words", 0]["words-recorder", 0].clear_cache()
    network["words", 0]["words-recorder", 0].variables = {"n.v": [], "n.fired": []}

    network.simulate_iterations(len(stream_i), measure_block_time=False)
    raster_plots(network, ngs=["letters", "words"])


if __name__ == "__main__":
    main()
