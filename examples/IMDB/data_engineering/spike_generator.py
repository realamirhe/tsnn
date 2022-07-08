from pandas import DataFrame

from examples.IMDB.data_engineering.test_train_dataset import test_train_dataset
from examples.IMDB.data_engineering.words_extractor import extract_words
from src.helpers.spikes import entity2spike


def words2spikes(df: DataFrame, common_words):
    words_stream_spikes = []
    oov_word = entity2spike("UNK", common_words)
    words_spikes_gap = [oov_word] * 2
    for idx, row in df.iterrows():
        for word in row["review"].split(" "):
            words_stream_spikes.append(entity2spike(word, common_words))
            words_stream_spikes.extend(words_spikes_gap)

    return words_stream_spikes


if __name__ == "__main__":
    (train_df, _) = test_train_dataset(train_size=50)
    common_words = extract_words(train_df)
    words_stream = words2spikes(train_df, common_words)
    print(words_stream)
    print("---")
