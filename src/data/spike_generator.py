import random

import numpy as np

from src.configs.corpus_config import letters, words, words_spacing_gap
from src.data.corpus_generator import gen_corpus


def spike_stream_i(char):
    spikes = np.zeros(len(letters), dtype=int)
    if char in letters:
        spikes[letters.index(char)] = 1
    return spikes


def get_data(size, prob=0.7, words_size=3):
    corpus = gen_corpus(
        size,
        prob,
        min_length=words_size,
        max_length=words_size,
        no_common_chars=False,
        letters_to_use=letters,
        words_to_use=words,
    )

    random.shuffle(corpus)

    sparse_gap = " " * words_spacing_gap
    joined_corpus = sparse_gap.join(corpus) + sparse_gap
    stream_i = [spike_stream_i(char) for char in joined_corpus]
    stream_j = []

    empty_spike = np.empty(len(words))
    empty_spike[:] = np.NaN

    # NOTE: 🚀 it seems that shifting all spikes won't chane the flow, but has more neuro-scientific effects
    for word in corpus:
        stream_j.extend((empty_spike for _ in word))

        # First space character after hole word
        word_spike = np.zeros(len(words), dtype=bool)
        if word in words:
            word_spike[words.index(word)] = 1
        stream_j.append(word_spike)

        stream_j.extend((empty_spike for _ in range(words_spacing_gap - 1)))

    if len(stream_i) != len(stream_j):
        raise AssertionError("stream length mismatch")

    return stream_i, stream_j, joined_corpus
