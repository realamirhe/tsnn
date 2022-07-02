import numpy as np

from PymoNNto import NeuronGroup, Network, SynapseGroup
from src.core.environement.dopamine import DopamineEnvironment
from src.core.learning.delay import SynapseDelay
from src.core.learning.stdp import SynapsePairWiseSTDP
from src.core.neurons.neurons import StreamableLIFNeurons
from src.core.neurons.trace import TraceHistory
from src.helpers.network import EpisodeTracker


def make_single_char_spikes(corpus, selected_character):
    spikes = []
    for char in corpus:
        spike = np.zeros(1, dtype=int)
        if char == selected_character:
            spike[0] = 1
        spikes.append(spike)
    return spikes


def main():
    network = Network()
    i_corpus = "i  "
    j_corpus = " j "
    DopamineEnvironment.set(1)
    connection_delay = 1
    lif_base = {
        "v_rest": -65,
        "v_reset": -65,
        "threshold": -52,
        "dt": 1.0,
        "R": 3,
        "tau": 3,
    }
    max_delay = 2
    i_stream = make_single_char_spikes(i_corpus, "i")
    j_stream = make_single_char_spikes(j_corpus, "j")
    i_ng = NeuronGroup(
        net=network,
        tag="i",
        size=1,
        behaviour={
            1: StreamableLIFNeurons(
                tag="i:force-spike", stream=i_stream, joined_corpus=i_corpus, **lif_base
            ),
            2: TraceHistory(max_delay=max_delay),
        },
    )
    j_ng = NeuronGroup(
        net=network,
        tag="j",
        size=1,
        behaviour={
            1: StreamableLIFNeurons(
                tag="j:force-spike", stream=j_stream, joined_corpus=j_corpus, **lif_base
            ),
            2: TraceHistory(max_delay=max_delay),
        },
    )

    SynapseGroup(
        net=network,
        src=i_ng,
        dst=j_ng,
        tag="GLUTAMATE",
        behaviour={
            # NOTE: 🚀 use max_delay to 4 and use_shared_weights=True
            1: SynapseDelay(
                tag="delay",
                max_delay=max_delay,
                mode="random",
                use_shared_weights=False,
            ),
            8: SynapsePairWiseSTDP(
                tag="stdp",
                tau_plus=4.0,
                tau_minus=4.0,
                a_plus=0.2,  # 0.02
                a_minus=-0.1,  # 0.01
                dt=1.0,
                w_min=0,
                # ((thresh - reset) / (3=characters) + epsilon) 4.33+eps
                w_max=np.round(
                    (lif_base["threshold"] - lif_base["v_rest"]) / 1  # j character size
                    + 2,  # epsilon: delay epsilon increase update, reduce full stimulus by tiny amount
                    decimals=1,
                ),
                min_delay_threshold=1,
                weight_decay=1,
                weight_update_strategy=None,
                stdp_factor=0.5,
                delay_factor=0.1,  # episode increase
            ),
        },
    )
    print("SynapseGroup", SynapseGroup)
    network.initialize(info=False)

    for _ in range(1):
        EpisodeTracker.update()
        network.iteration = 0
        network.simulate_iterations(len(i_corpus))


if __name__ == "__main__":
    main()
