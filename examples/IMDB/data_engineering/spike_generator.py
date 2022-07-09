from pandas import DataFrame

from examples.IMDB.config import words_spacing_gap
from examples.IMDB.data_engineering.test_train_dataset import test_train_dataset
from examples.IMDB.data_engineering.words_extractor import extract_words
from src.helpers.spikes import entity2spike

"""
NOTE: joined_corpus are somehow redundant and is generated for debug
      also WORDS_GAP are hard-coded 
"""
WORDS_GAP = words_spacing_gap


def words2spikes(df: DataFrame, common_words):
    words_stream_spikes = []
    oov_word = entity2spike("UNK", common_words)
    words_spikes_gap = [oov_word] * WORDS_GAP
    for idx, row in df.iterrows():
        for word in row["review"].split(" "):
            words_stream_spikes.append(entity2spike(word, common_words))
            words_stream_spikes.extend(words_spikes_gap)

    return words_stream_spikes


def sentence2spikes(df: DataFrame):
    sentence_stream_spikes = {}
    stream_sentence_idx = -1
    words_spikes_gap = WORDS_GAP
    for idx, row in df.iterrows():
        words_count = len(row["review"].split(" ")) * (1 + words_spikes_gap)
        stream_sentence_idx += words_count
        sentence_stream_spikes[stream_sentence_idx] = row["sentiment"]

    return sentence_stream_spikes


def joined_corpus_maker(df: DataFrame):
    reviews = []
    words_spikes_gap = [" "] * WORDS_GAP
    for idx, row in df.iterrows():
        for word in row["review"].split(" "):
            reviews.append(word)
            reviews.extend(words_spikes_gap)

    return reviews


if __name__ == "__main__":
    (train_df, _) = test_train_dataset(train_size=50)
    common_words = extract_words(train_df)
    words_stream = words2spikes(train_df, common_words)
    print(words_stream)
    print("---")
    print(joined_corpus_maker(train_df))
