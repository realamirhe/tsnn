import string
from os import getcwd
from os.path import join

import numpy as np

with open(join(getcwd(), "corpus/common_words.npy"), "rb") as file:
    common_words = np.load(file)[:100]

maximum_length_word = max(common_words, key=len)
minimum_length_word = min(common_words, key=len)
average_length = sum(map(len, common_words)) // len(common_words)
letters = string.ascii_lowercase + string.digits

max_delay = len(maximum_length_word) + 1
words_spacing_gap = 10
words_capture_window_size = words_spacing_gap + len(maximum_length_word)


def corpus_config():
    print(
        f"""
        {len(common_words)} words has been loaded.
        The longest words is "{maximum_length_word}" which is {len(maximum_length_word)} char long.
        The shortest words is "{minimum_length_word}" which is {len(minimum_length_word)} char long.
        The words are roughly "{average_length}" character long on average.
        The network (letters => words) layer instantiate with maximum delay of {max_delay}.
        Words will decay roughly in {words_spacing_gap} timestep.
        Words capturing window used in homeostasis is {words_capture_window_size}.
        Network Storages are in the shape of the followings:
         - synapse.W, synapse.delay ({len(common_words)}x{len(letters)})
         - letters.trace            ({len(letters)}x{max_delay + 1})
         - words.trace              ({len(common_words)}x{max_delay + 1})
        """
    )

    return {
        "max_delay": max_delay,
        "letters": letters,
        "language": letters + " ",  # out_of_vocab_separator
        "words": common_words,
        "words_spacing_gap": words_spacing_gap,
        "words_capture_window_size": words_capture_window_size,
        "UNK": "UNK",
    }


if __name__ == "__main__":
    corpus_config()
