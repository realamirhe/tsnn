import numpy as np

from PymoNNto import SynapseGroup, Recorder, NeuronGroup, Network
from src.core.learning.delay import SynapseDelay
from src.core.learning.reinforcement import Supervisor
from src.core.learning.stdp import SynapsePairWiseSTDP
from src.core.metrics.metrics import Metrics
from src.core.neurons.neurons import StreamableLIFNeurons
from src.core.stabilizer.activity_base_homeostasis import ActivityBaseHomeostasis
from src.core.stabilizer.winner_take_all import WinnerTakeAll
from src.data.constants import letters, words
from src.data.spike_generator import get_data
from src.helpers.base import reset_random_seed, behaviour_generator
from src.helpers.network import FeatureSwitch

reset_random_seed(1230)


# ================= NETWORK  =================
def main():
    network = Network()
    stream_i_train, stream_j_train, corpus_train = get_data(1000, prob=0.9)
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
                StreamableLIFNeurons(
                    **lif_base, has_long_term_effect=True, capture_old_v=True
                ),
                ActivityBaseHomeostasis(
                    tag="homeostasis",
                    min_activity=7,
                    max_activity=8,
                    window_size=100,
                    updating_rate=0.001,
                    activity_rate=7.5,  # can be calculated from the corpus
                ),
                # Hamming-distance
                # distance 0 => dopamine release
                # Fire() => dopamine_decay should reset a word 1  by at last 3(max delay) time_steps
                # differences must become 0 after some time => similar
                WinnerTakeAll(),
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
        behaviour={
            1: SynapseDelay(max_delay=4, mode="random", use_shared_weights=False),
            5: SynapsePairWiseSTDP(
                tag="stdp",
                tau_plus=4.0,
                tau_minus=4.0,
                a_plus=0.01,
                a_minus=-0.01,
                dt=1.0,
                w_min=0,
                w_max=4.5,  # ((thresh - reset) / (3=characters) + epsilon) 4.33+eps
                stdp_factor=0.1,
                delay_epsilon=0.15,
                weight_decay=0.0,
                stimulus_scale_factor=0.01,
            ),
        },
    )

    network.initialize()

    features = FeatureSwitch(network, ["lif", "supervisor", "metrics", "spike-rate"])
    features.switch_train()

    """ TRAINING """
    epochs = 10
    for episode in range(epochs):
        network.iteration = 0
        network.simulate_iterations(len(stream_i_train))
        weights = network.SynapseGroups[0].W
        print(
            f"episode={episode} sum={np.sum(weights):.1f}, max={np.max(weights):.1f}, min={np.min(weights):.1f}"
        )
        network["letters-recorder", 0].reset()
        network["words-recorder", 0].reset()
        network["metrics:train", 0].reset()

    """ TESTING """
    # features.switch_test()
    # # Hacky integration, preventing another weight copy!
    # network["stdp", 0].recording = False
    # network.iteration = 0
    # network["words-recorder", 0].clear_cache()
    # network["words-recorder", 0].variables = {"n.v": [], "n.fired": []}
    # network.simulate_iterations(len(stream_i_test))


if __name__ == "__main__":
    main()