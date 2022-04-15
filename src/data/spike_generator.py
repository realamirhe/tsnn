import random

import numpy as np

from src.data.constants import letters, words
from src.data.corpus_generator import gen_corpus


def spike_stream_i(char):
    spikes = np.zeros(len(letters), dtype=int)
    if char in letters:
        spikes[letters.index(char)] = 1
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
    sparse_gap = " " * 1  # TODO: When we should use more sparsity gap
    joined_corpus = sparse_gap.join(corpus) + sparse_gap
    stream_i = [spike_stream_i(char) for char in joined_corpus]
    stream_j = []

    empty_spike = np.empty(len(words))
    empty_spike[:] = np.NaN

    # NOTE: 🚀 it seems that shifting all spikes won't chane the flow, but has more neuro-scientific effects
    # uncomment line 39 and comment line 49-50 to see the difference
    for word in corpus:
        for _ in word:
            # for _ in range(len(word) - 1):
            stream_j.append(empty_spike)

        word_spike = np.zeros(len(words), dtype=bool)
        if word in words:
            word_index = words.index(word)
            word_spike[word_index] = 1
        stream_j.append(word_spike)  # spike when see hole word!

        # for _ in sparse_gap:
        #     stream_j.append(empty_spike)  # space between words

    assert len(stream_i) == len(stream_j), "stream length mismatch"
    return stream_i, stream_j, corpus
