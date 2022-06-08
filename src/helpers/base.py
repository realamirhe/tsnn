import random

import numpy as np

from src.configs import corpus_config


def reset_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def behaviour_generator(behaviours):
    return {index + 1: behaviour for index, behaviour in enumerate(behaviours)}


def selected_neurons_from_words():
    rows = np.array(
        [[i for _ in word] for i, word in enumerate(corpus_config.words)]
    ).flatten()

    cols = np.array(
        [
            [corpus_config.letters.index(letter) for letter in word]
            for word in corpus_config.words
        ]
    ).flatten()

    return rows, cols
