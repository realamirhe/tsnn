import random
import string
from typing import List

import numpy as np

from src.libs.helper import reset_random_seed

reset_random_seed()

LETTERS = string.ascii_lowercase
WORDS = ["abc", "omn"]

# ================= CORPUS AND DATA  =================
WORDS = [word.lower() for word in WORDS]
# "words must be non overlapping for first stage"
assert len(set(WORDS[0]).intersection(WORDS[1])) == 0


def gen_corpus(
        size=100, prob=0.5, min_length=1, max_length=10, no_common_chars: bool = False
) -> List[str]:
    """
        Generate a corpus of random words. contains learnable words within
    """
    corpus: List[str] = []
    valid_letters = LETTERS
    if no_common_chars:
        words_character = set("".join(WORDS))
        valid_letters = set(valid_letters).symmetric_difference(words_character)
        valid_letters = "".join(valid_letters)

    for i in range(size):
        if random.random() < prob:
            word = random.choice(WORDS)
        else:
            word_length = random.randint(min_length, max_length)
            word = "".join(random.choice(valid_letters) for _ in range(word_length))
        corpus.append(word)
    return corpus


def char2spike(char: str) -> np.array:
    """
        Convert a character to a spike vector
        :param char: character to convert
        :return: spike vector
    """
    spike = np.zeros(len(LETTERS))
    if char != " ":
        spike[LETTERS.index(char)] = 1
    return spike


def get_word_label(word) -> List[int]:
    # +1 for the " " which is then concatenated to the each intermediate word
    word_labels = [-1] * (len(word) + 1)
    if word in WORDS:
        word_labels[-2] = WORDS.index(word)  # -2 is position of last seen character
    # print(word, word_labels)
    return tuple(word_labels)


# ================= DATA GENREATOR  =================
def stream_generator_for_character(target_char, noise, size):
    stream_characters = [target_char] + list(LETTERS.replace(target_char, ""))
    stream_characters = np.random.choice(stream_characters, size)
    for char_index in range(stream_characters.shape[0]):
        if np.random.random() > noise:
            stream_characters[char_index] = target_char
    return stream_characters


# ================= GENERIC DATA  =================
CORPUS = gen_corpus(
    size=40, prob=0.8, no_common_chars=False, min_length=1, max_length=3
)
labels = [get_word_label(word) for word in CORPUS]
labels = np.concatenate(labels).ravel()
CORPUS = " ".join(CORPUS)

# ================= CONSUMED DATA =================
input_data = [char2spike(char) for char in CORPUS]
letters_input_data = np.stack(input_data).astype(bool)

if __name__ == "__main__":
    # print(CORPUS)
    # print(len(CORPUS))
    # print(labels.shape)
    # print(labels.dtype)
    # print(labels)
    print(stream_generator_for_character(target_char="i", noise=0.1, size=5))
    # [0, 0, 1] w
    # [0 , 0, w]
    # [1, 0, 1]  * w / np.sum()
