from PymoNNto import NeuronGroup
from examples.archive.IMDB_slow.configurations.corpus_config import corpus_stream_config
from examples.archive.IMDB_slow.configurations.network_config import letters, max_delay
from examples.archive.IMDB_slow.configurations.neurons_config import lif_base_config
from src.core.neurons.neurons import StreamableLIFNeurons
from src.core.neurons.trace import TraceHistory


def letter_ng(network):
    corpus = corpus_stream_config("corpus")
    stream_input = corpus_stream_config("stream_i")
    lif_base = lif_base_config()

    return NeuronGroup(
        net=network,
        tag="letters",
        size=len(letters),
        behaviour={
            1: StreamableLIFNeurons(
                tag="lif:train", stream=stream_input, joined_corpus=corpus, **lif_base
            ),
            2: TraceHistory(max_delay=max_delay),
        },
    )
