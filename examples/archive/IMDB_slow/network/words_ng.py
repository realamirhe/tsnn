from PymoNNto import NeuronGroup
from examples.archive.IMDB_slow.configurations.corpus_config import corpus_stream_config
from examples.archive.IMDB_slow.configurations.network_config import (
    max_delay,
    common_words,
    words_capture_window_size,
)
from examples.archive.IMDB_slow.configurations.neurons_config import lif_base_config
from src.core.learning.reinforcement import Supervisor
from src.core.metrics.metrics import Metrics
from src.core.neurons.current import CurrentStimulus
from src.core.neurons.neurons import StreamableLIFNeurons
from src.core.neurons.trace import TraceHistory
from src.core.stabilizer.activity_base_homeostasis import ActivityBaseHomeostasis
from src.core.stabilizer.winner_take_all import WinnerTakeAll


def words_ng(network, homeostasis_window_size=1000):
    stream_i = corpus_stream_config("stream_i")
    stream_words = corpus_stream_config("stream_words")
    lif_base = lif_base_config()
    lif_base["v_reset"] = (
        lif_base["v_reset"] - (lif_base["R"] / lif_base["tau"]) * max_delay
    )

    return NeuronGroup(
        net=network,
        tag="words",
        size=len(common_words),
        behaviour={
            2: CurrentStimulus(
                adaptive_noise_scale=0.9,
                noise_scale_factor=0.1,
                stimulus_scale_factor=1,
                synapse_lens_selector=["GLUTAMATE", 0],
            ),
            # NOTE: can enable reset factory if you want
            3: StreamableLIFNeurons(
                **lif_base,
                has_long_term_effect=True,
                capture_old_v=True,
            ),
            4: TraceHistory(max_delay=max_delay),
            5: ActivityBaseHomeostasis(
                tag="homeostasis",
                window_size=homeostasis_window_size,
                updating_rate=0.01,
                # NOTE: the homeostasis ratio rate is not proper, ratio for frequent corpus slice
                activity_rate=homeostasis_window_size / words_capture_window_size,
            ),
            6: WinnerTakeAll(),
            7: Supervisor(
                tag="supervisor:train",
                dopamine_decay=1 / (max_delay + 1),
                outputs=stream_words,
            ),
            9: Metrics(
                tag="metrics:train",
                words=common_words,
                outputs=stream_words,
                episode_iterations=len(stream_i),
            ),
        },
    )
