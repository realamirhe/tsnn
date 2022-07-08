import functools
from os import getcwd
from os.path import join

import numpy as np
import pandas as pd


@functools.lru_cache(maxsize=1)
def corpus_stream_config(key):
    if key == "stream_i":
        with open(join(getcwd(), "corpus/streams_demo.npz"), "rb") as file:
            return np.load(file)["stream_i"][:400]
    if key == "corpus":
        with open(join(getcwd(), "corpus/streams_demo.npz"), "rb") as file:
            return str(np.load(file)["corpus"])[:400]

    if key == "stream_words":
        return pd.read_pickle(join(getcwd(), "corpus/stream_words.npy"))

    if key == "stream_sentences":
        return pd.read_pickle(join(getcwd(), "corpus/stream_sentences.npy"))
