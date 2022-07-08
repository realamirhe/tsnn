from tqdm import tqdm

from PymoNNto import Network
from examples.archive.IMDB_slow.configurations.corpus_config import corpus_stream_config
from examples.archive.IMDB_slow.network.letters_ng import letter_ng
from examples.archive.IMDB_slow.network.letters_words_sg import letters_words_sg
from examples.archive.IMDB_slow.network.words_ng import words_ng

# TODO: fix the convention of new stream type
# TODO: add the convergence condition for early stop
# TODO: add the third layer of the model for neural population and classification
from src.helpers.network import EpisodeTracker


def network():
    network = Network()
    letters = letter_ng(network)
    words = words_ng(network)
    letters_words_sg(network, letters, words)

    network.initialize(info=False)
    simulation_iterations = len(corpus_stream_config("stream_i"))
    """ TRAINING """
    for _ in tqdm(range(150), "Learning"):
        EpisodeTracker.update()
        network.iteration = 0
        # TODO: fix time.time() from git
        network.simulate_iterations(simulation_iterations, measure_block_time=False)
        for tag in ["metrics:train"]:
            network[tag, 0].reset()


if __name__ == "__main__":
    network()
