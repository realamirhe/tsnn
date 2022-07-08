import random

import numpy as np
from tqdm import tqdm

from src.configs.corpus_config import letters, words, words_spacing_gap
from src.data.corpus_generator import gen_corpus


def spike_stream_i(char):
    spikes = np.zeros(len(letters), dtype=int)
    if char in letters:
        spikes[letters.index(char)] = 1
    return spikes


"""

def joined_corpus_generator(corpus: List[str], has_noise=False) -> str:
    if words_spacing_gap < 2 or not has_noise:
        sparse_gap = " " * words_spacing_gap
        return sparse_gap.join(corpus) + sparse_gap

    space_seen_probability = 0.5
    p = np.ones(len(letters) + 1)
    p *= (1 - space_seen_probability) / (p.size - 1)
    p[0] = space_seen_probability

    possible_noise = [" "] + list(letters)
    return "".join(
        [
            word
            + " "
            + "".join(np.random.choice(possible_noise, words_spacing_gap - 2, p=p))
            + " "
            for word in corpus
        ]
    )
"""


def adaptive_noise_corpus(size, space_seen_probability=None):
    if words_spacing_gap < 2 or space_seen_probability is None:
        sparse_gap = " " * words_spacing_gap
        return [sparse_gap] * size

    noise = np.random.random((size, words_spacing_gap))
    noise_scale = np.linspace(1, 0, size)  # adaptive noise
    noise = noise * noise_scale[:, np.newaxis]
    noise[:, [0, -1]] = 0
    noise[noise <= space_seen_probability] = 0

    noise = (noise * 1000).astype(int) % (len(letters) + 1)
    numpy_letters = np.array(list(" " + letters))
    noise = numpy_letters[noise]
    noise = noise.tolist()
    noise = ("".join(noise_gap) for noise_gap in noise)
    return noise


def get_data(size, prob=0.7, words_size=3):
    # TODO: generate corpus from the dataset
    generated_words = gen_corpus(
        size,
        prob,
        min_length=words_size,
        max_length=words_size,
        no_common_chars=False,
        letters_to_use=letters,
        words_to_use=words,
    )

    random.shuffle(generated_words)

    stream_words = {}
    stream_words_idx = 0

    did_not_spiked = -1
    # NOTE: 🚀 it seems that shifting all spikes won't chane the flow, but has more neuro-scientific effects
    noise = adaptive_noise_corpus(len(generated_words), space_seen_probability=0.9)
    corpus = []
    for word, noise_gap in tqdm(
        zip(generated_words, noise),
        desc="generating spikes...",
        total=len(generated_words),
    ):
        noised_adapted_word = word + noise_gap
        corpus.append(noised_adapted_word)

        seen_word_index = words.index(word) if word in words else did_not_spiked
        stream_words[stream_words_idx + len(word)] = seen_word_index
        stream_words_idx += len(noised_adapted_word)

    joined_corpus = "".join(corpus)
    stream_i = [spike_stream_i(char) for char in joined_corpus]

    return stream_i, stream_words, joined_corpus
