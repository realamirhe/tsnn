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
    stream_i_train, stream_j_train, corpus_train = get_data(1000, prob=0.6)
    stream_i_test, stream_j_test, corpus_test = get_data(1000, prob=0.6)

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
                    tag="lif:train",
                    stream=stream_i_train,
                    corpus=corpus_train,
                    **lif_base,
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
        behaviour={
            1: StreamableLIFNeurons(
                **lif_base, has_long_term_effect=True, capture_old_v=True,
            ),
            2: ActivityBaseHomeostasis(
                tag="homeostasis",
                window_size=100,
                # NOTE: making updating_rate adaptive is not useful, because we are training model multiple time
                # so long term threshold must be set within one of these passes. It is useful for faster convergence
                updating_rate=0.01,
                activity_rate=15,
                # window_size = 100 character every word has 3 character + space, so we roughly got 25
                # spaced words per window; 0.6 of words are desired so 25*0.6 = 15 are expected to spike
                # in each window (15 can be calculated from the corpus)
            ),
            # Hamming-distance
            # distance 0 => dopamine release
            # Fire() => dopamine_decay should reset a word 1  by at last 3(max delay) time_steps
            # differences must become 0 after some time => similar
            3: WinnerTakeAll(),
            4: Supervisor(
                tag="supervisor:train", dopamine_decay=1 / 3, outputs=stream_j_train
            ),
            5: Supervisor(
                tag="supervisor:test", dopamine_decay=1 / 3, outputs=stream_j_test
            ),
            7: Metrics(
                tag="metrics:train",
                words=words,
                outputs=stream_j_train,
                corpus=corpus_train,
            ),
            8: Metrics(
                tag="metrics:test",
                words=words,
                outputs=stream_j_test,
                corpus=corpus_test,
            ),
            9: Recorder(tag="words-recorder", variables=["n.v", "n.fired"]),
        },
    )

    SynapseGroup(
        net=network,
        src=letters_ng,
        dst=words_ng,
        tag="GLUTAMATE",
        behaviour={
            # NOTE: 🚀 use max_delay to 4 and use_shared_weights=True
            1: SynapseDelay(max_delay=3, mode="random", use_shared_weights=False),
            6: SynapsePairWiseSTDP(
                tag="stdp",
                tau_plus=4.0,
                tau_minus=4.0,
                a_plus=0.01,
                a_minus=-0.01,
                dt=1.0,
                w_min=0,
                # ((thresh - reset) / (3=characters) + epsilon) 4.33+eps
                w_max=np.round(
                    (lif_base["threshold"] - lif_base["v_rest"])
                    / np.average(list(map(len, words))),
                    decimals=1,
                ),
                stdp_factor=0.1,
                delay_factor=1e1,  # episode increase
                min_delay_threshold=0.15,
                weight_decay=0,
                stimulus_scale_factor=1,
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
        print(f"{episode + 1}::long term threshold", network.NeuronGroups[1].threshold)
        # print("final delay", network.SynapseGroups[0].delay)
        network["letters-recorder", 0].reset()
        network["words-recorder", 0].reset()
        network["metrics:train", 0].reset()
        # raster_plots(network)
        # voltage_plots(network)

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
