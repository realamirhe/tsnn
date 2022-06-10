import cProfile
import functools
import io
import pstats
import random

import numpy as np

from src.configs import corpus_config
from src.configs.feature_flags import enabled_c_profiler


def reset_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def behaviour_generator(behaviours):
    return {index + 1: behaviour for index, behaviour in enumerate(behaviours)}


@functools.lru_cache(maxsize=1)
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


def c_profiler(func):
    if not enabled_c_profiler:
        return func

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()

        func(*args, **kwargs)

        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
        ps.print_stats()

        with open("report.txt", "w+") as f:
            f.write(s.getvalue())

    return wrapper


if __name__ == "__main__":
    print(selected_neurons_from_words())
    print(selected_neurons_from_words())
    print(selected_neurons_from_words())
