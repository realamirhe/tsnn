import string
from collections import namedtuple

corpus = dict(
    gap=7,
    language=string.ascii_lowercase + " ",
    letters=string.ascii_lowercase,
    words=["abc", "omn"],
)
corpus = namedtuple("Corpus", corpus.keys())(**corpus)
