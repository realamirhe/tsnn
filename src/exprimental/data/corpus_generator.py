import random
from typing import List

from src.exprimental.data.constants import letters, words


def gen_corpus(
    size=100,
    prob=0.5,
    min_length=1,
    max_length=10,
    no_common_chars: bool = False,
    letters_to_use: str = letters,
    words_to_use: List[str] = words,
) -> List[str]:
    """
        Generate a corpus of random words. contains learnable words within
    """
    corpus: List[str] = []
    valid_letters = letters_to_use
    if no_common_chars:
        words_character = set("".join(words_to_use))
        valid_letters = set(valid_letters).symmetric_difference(words_character)
        valid_letters = "".join(valid_letters)

    for i in range(size):
        if random.random() < prob:
            word = random.choice(words_to_use)
        else:
            word_length = random.randint(min_length, max_length)
            word = "".join(random.choice(valid_letters) for _ in range(word_length))
        corpus.append(word)
    return corpus