import numpy as np
from tqdm import tqdm

from PymoNNto import SynapseGroup, Recorder, NeuronGroup, Network
from src.configs import corpus_config, feature_flags
from src.core.learning.delay import SynapseDelay
from src.core.learning.reinforcement import Supervisor
from src.core.learning.stdp import SynapsePairWiseSTDP
from src.core.metrics.metrics import Metrics
from src.core.neurons.current import CurrentStimulus
from src.core.neurons.neurons import StreamableLIFNeurons
from src.core.neurons.trace import TraceHistory
from src.core.stabilizer.activity_base_homeostasis import ActivityBaseHomeostasis
from src.core.stabilizer.winner_take_all import WinnerTakeAll
from src.data.spike_generator import get_data
from src.helpers.base import reset_random_seed
from src.helpers.network import FeatureSwitch, EpisodeTracker

reset_random_seed(1231)
max_delay = 3


# ================= NETWORK  =================
def main():
    network = Network()
    stream_i_train, stream_j_train, joined_corpus = get_data(1000, prob=0.9)

    lif_base = {
        "v_rest": -65,
        "v_reset": -65,
        "threshold": -52,
        "dt": 1.0,
        "R": 3,
        "tau": 3,
    }

    letters_ng = NeuronGroup(
        net=network,
        tag="letters",
        size=len(corpus_config.letters),
        behaviour={
            1: StreamableLIFNeurons(
                tag="lif:train",
                stream=stream_i_train,
                joined_corpus=joined_corpus,
                **lif_base,
            ),
            2: TraceHistory(max_delay=max_delay),
            3: Recorder(tag="letters-recorder", variables=["n.v", "n.fired"]),
        },
    )

    words_ng = NeuronGroup(
        net=network,
        tag="words",
        size=len(corpus_config.words),
        behaviour={
            2: CurrentStimulus(
                adaptive_noise_scale=0.9,
                noise_scale_factor=1,
                stimulus_scale_factor=1,
                synapse_lens_selector=["GLUTAMATE", 0],
            ),
            3: StreamableLIFNeurons(
                **(
                    lif_base
                    if feature_flags.enable_reset_factory
                    else {
                        **lif_base,
                        "v_reset": -65 - (lif_base["R"] / lif_base["tau"]) * max_delay,
                    }
                ),
                has_long_term_effect=True,
                capture_old_v=True,
            ),
            4: TraceHistory(max_delay=max_delay),
            5: ActivityBaseHomeostasis(
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
            6: WinnerTakeAll(),
            7: Supervisor(
                tag="supervisor:train",
                dopamine_decay=1 / max_delay,
                outputs=stream_j_train,
            ),
            9: Metrics(
                tag="metrics:train",
                words=corpus_config.words,
                outputs=stream_j_train,
            ),
            11: Recorder(tag="words-recorder", variables=["n.v", "n.fired"]),
        },
    )

    SynapseGroup(
        net=network,
        src=letters_ng,
        dst=words_ng,
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
                a_plus=0.02,
                a_minus=-0.01,
                dt=1.0,
                w_min=0,
                # ((thresh - reset) / (3=characters) + epsilon) 4.33+eps
                w_max=np.round(
                    (lif_base["threshold"] - lif_base["v_rest"])
                    / np.average(list(map(len, corpus_config.words))),
                    decimals=1,
                ),
                min_delay_threshold=1,  # 0.15,
                weight_decay=0,
                weight_update_strategy=None,
                stdp_factor=0.5,
                delay_factor=1,  # episode increase
            ),
        },
    )
    network.initialize(info=False)

    features = FeatureSwitch(network, ["lif", "supervisor", "metrics", "spike-rate"])
    features.switch_train()

    """ TRAINING """
    epochs = 50
    for _ in tqdm(range(epochs), "Learning"):
        EpisodeTracker.update()
        network.iteration = 0
        network.simulate_iterations(len(stream_i_train))
        for tag in ["letters-recorder", "words-recorder", "metrics:train"]:
            network[tag, 0].reset()


if __name__ == "__main__":
    main()
