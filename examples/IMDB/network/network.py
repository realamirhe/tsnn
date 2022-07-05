from PymoNNto import Network
from examples.IMDB.configurations.data_config import get_data
from examples.IMDB.configurations.network_config import corpus_config


def network():
    network = Network()
    stream_i_train, stream_j_train, joined_corpus = get_data(1000, prob=0.9)


if __name__ == "__main__":
    corpus_config()
