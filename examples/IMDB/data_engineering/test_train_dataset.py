from os.path import join, dirname

import pandas as pd


def test_train_dataset(train_size=100, test_size=20):
    df = pd.read_csv(join(dirname(__file__), f"../corpus/IMDB_preprocessed.csv"))
    per_class_chunk_size = train_size + test_size

    train_test_positive_class = df[df["sentiment"] == 1].sample(
        n=per_class_chunk_size, replace=False, random_state=42
    )

    train_test_negative_class = df[df["sentiment"] == 0].sample(
        n=per_class_chunk_size, replace=False, random_state=42
    )

    train_set = [
        train_test_positive_class.iloc[:train_size],
        train_test_negative_class.iloc[:train_size],
    ]

    test_set = [
        train_test_positive_class.iloc[:test_size],
        train_test_negative_class.iloc[:test_size],
    ]

    return pd.concat(train_set), pd.concat(test_set)


if __name__ == "__main__":
    train, test = test_train_dataset(train_size=50, test_size=10)
    print(train.shape, test.shape)
