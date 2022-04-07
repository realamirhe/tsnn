import random

import numpy as np

from exprimental.data.corpus_generator import gen_corpus
from exprimental.data.constants import letters, words


def spike_stream_i(char):
    spikes = np.zeros(len(letters), dtype=int)
    if char in letters:
        # TODO: this must not be hardcoded, best is bool and 1
        spikes[letters.index(char)] = 90
    return spikes


def get_data(size, prob=0.7):
    corpus = gen_corpus(
        size,
        prob,
        min_length=3,
        max_length=3,
        no_common_chars=False,
        letters_to_use=letters,
        words_to_use=words,
    )
    random.shuffle(corpus)
    sparse_gap = " " * 7
    joined_corpus = sparse_gap.join(corpus) + sparse_gap
    stream_i = [spike_stream_i(char) for char in joined_corpus]
    stream_j = []

    empty_spike = np.empty(len(words))
    empty_spike[:] = np.NaN

    for word in corpus:
        for char in range(len(word) - 1):
            stream_j.append(empty_spike)

        word_spike = np.zeros(len(words), dtype=bool)
        if word in words:
            word_index = words.index(word)
            word_spike[word_index] = 1
        stream_j.append(word_spike)  # spike when see hole word!

        for _ in range(len(sparse_gap)):
            stream_j.append(empty_spike)  # space between words

    assert len(stream_i) == len(stream_j), "stream length mismatch"
    return stream_i, stream_j, corpus
