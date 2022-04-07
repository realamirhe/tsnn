import numpy as np

from PymoNNto import SynapseGroup, Recorder, NeuronGroup, Network
from src.exprimental.core.learning.delay import SynapseDelay
from src.exprimental.core.learning.reinforcement import Supervisor
from src.exprimental.core.learning.stdp import SynapsePairWiseSTDP
from src.exprimental.core.metrics.metrics import Metrics
from src.exprimental.core.neurons.neurons import StreamableLIFNeurons
from src.exprimental.core.stabilizer.homeostasis import Homeostasis
from src.exprimental.core.stabilizer.spike_rate import SpikeRate
from src.exprimental.data.constants import letters, words
from src.exprimental.data.spike_generator import get_data
from src.exprimental.helpers.base import reset_random_seed, behaviour_generator

reset_random_seed(42)


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
                StreamableLIFNeurons(
                    tag="lif:train", stream=stream_i_train, **lif_base
                ),
                StreamableLIFNeurons(tag="lif:test", stream=stream_i_test, **lif_base),
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
                StreamableLIFNeurons(**lif_base, has_long_term_effect=True),
                Homeostasis(
                    tag="homeostasis",
                    max_ta=-55,
                    min_ta=-70,
                    eta_ip=0.1,
                    has_long_term_effect=True,
                ),
                # Hamming-distance
                # differences must become 0 after some time => similar
                SpikeRate(
                    tag="spike-rate:train", interval_size=5, outputs=stream_j_train
                ),
                SpikeRate(
                    tag="spike-rate:train", interval_size=5, outputs=stream_j_test
                ),
                # Fire(),
                #  dopamine_decay should reset a word 1  by at last 3(max delay) time_steps
                # distance 0 => dopamine release
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
                    weight_decay=0.0,
                ),
            ]
        ),
    )

    network.initialize()
    train_mechanising = [
        "lif:train",
        "supervisor:train",
        "metrics:train",
        "spike-rate:train",
    ]
    test_mechanising = [
        "lif:test",
        "supervisor:test",
        "metrics:test",
        "spike-rate:test",
    ]

    network.activate_mechanisms(train_mechanising)
    network.deactivate_mechanisms(test_mechanising)
    epochs = 2
    for episode in range(epochs):
        network.iteration = 0
        network.simulate_iterations(len(stream_i_train))
        weights = network.SynapseGroups[0].W
        print(
            f"episode={episode} sum={np.sum(weights):.1f}, max={np.max(weights):.1f}, min={np.min(weights):.1f}"
        )
        # raster_plots(network, ngs=["words"])
        network["letters-recorder", 0].reset()
        network["words-recorder", 0].reset()
        network["metrics:train", 0].reset()
        # raster_plots(network, ngs=["letters"])
        # raster_plots(network, ngs=["words"])

    network.activate_mechanisms(test_mechanising)
    network.deactivate_mechanisms(train_mechanising)
    # Hacky integration, preventing another weight copy!
    network["stdp", 0].recording = False

    network.iteration = 0
    network["words-recorder", 0].clear_cache()
    network["words-recorder", 0].variables = {"n.v": [], "n.fired": []}
    network.simulate_iterations(len(stream_i_test))
    # raster_plots(network, ngs=["words"])


if __name__ == "__main__":
    main()
