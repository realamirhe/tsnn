from examples.archive.IMDB_slow.configurations.network_config import (
    words_spacing_gap,
    maximum_length_word,
)


def lif_base_config():
    return {
        "v_rest": -65,
        "v_reset": -65,
        "threshold": -55,
        "dt": 1.0,
        "R": 1,
        "tau": max(words_spacing_gap, len(maximum_length_word)),
    }
