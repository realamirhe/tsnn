import numpy as np
import pandas as pd

from examples.IMDB.configurations.network_config import (
    words_spacing_gap,
    letters,
    common_words,
)


def spike_stream_i(char):
    spikes = np.zeros(len(letters), dtype=int)
    if char in letters:
        spikes[letters.index(char)] = 1
    return spikes


def noised_gaped_word(word: str, space_seen_probability=None) -> str:
    if words_spacing_gap < 2 or space_seen_probability is None:
        sparse_gap = " " * words_spacing_gap
        return word + sparse_gap

    p = np.ones(len(letters) + 1)
    p *= (1 - space_seen_probability) / (p.size - 1)
    p[0] = space_seen_probability

    possible_noise = [" "] + list(letters)
    return "{} {} ".format(
        word, "".join(np.random.choice(possible_noise, words_spacing_gap - 2, p=p))
    )


def get_data(imdb_df):
    corpus = []
    stream_words = {}
    stream_sentences = {}
    stream_words_idx = 0
    stream_sentences_idx = 0
    for index, record in imdb_df.iterrows():
        for word in record["review"].split(" "):
            # possible adaptive noise
            noised_adapted_word = noised_gaped_word(word, space_seen_probability=0.9)
            corpus.append(noised_adapted_word)
            if word in common_words:
                seen_word_index = np.where(common_words == word)[0][0]
            else:
                # Possible map these to UNK word
                seen_word_index = -1
            stream_words[stream_words_idx + len(word)] = seen_word_index
            stream_words_idx += len(noised_adapted_word)
            stream_sentences_idx += len(noised_adapted_word)
        corpus.append(".")
        stream_sentences[stream_sentences_idx] = record["sentiment"]

    joined_corpus = "".join(corpus)
    stream_i = [spike_stream_i(char) for char in joined_corpus]

    return joined_corpus, (stream_i, stream_words, stream_sentences)


if __name__ == "__main__":
    df = pd.read_csv("../corpus/IMDB_preprocessed.csv")
    df = df.iloc[:100]
    corpus, streams = get_data(df)
    with open("../corpus/streams_demo.npz", "wb") as file:
        np.savez(
            file,
            corpus=corpus,
            stream_i=streams[0],
            stream_words=streams[1],
            stream_sentences=streams[2],
        )

    # df_train, df_test = train_test_split(df, test_size=0.2, random_state=2022)
    # corpus_test, streams_test = get_data(df_test)
    # corpus_train, streams_train = get_data(df_train)
