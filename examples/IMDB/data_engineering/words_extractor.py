from pandas import DataFrame

from examples.IMDB.data_engineering.test_train_dataset import test_train_dataset


# NOTE: words were pre-processed and the longest word was 17 character long
def extract_words(train_df: DataFrame, word_length_threshold=17):
    words = []
    for idx, row in train_df.iterrows():
        words.extend(
            filter(
                lambda word: len(word) <= word_length_threshold,
                row["review"].split(" "),
            )
        )

    return list(set(words))


if __name__ == "__main__":
    (train_df, _) = test_train_dataset(train_size=50)
    print(train_df.sentiment.value_counts())
    words = extract_words(train_df)
    print(len(words))
    print(words)
