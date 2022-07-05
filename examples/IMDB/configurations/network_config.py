import string

import numpy as np

with open("../corpus/common_words.npy", "rb") as file:
    common_words = np.load(file)

maximum_length_word = max(common_words, key=len)
minimum_length_word = min(common_words, key=len)
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
        The network (letters => words) layer instantiate with maximum delay of {max_delay}.
        Words will decay roughly in {words_spacing_gap} timestep.
        Words capturing window used in homeostasis is {words_capture_window_size}. 
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
