from os.path import join, dirname

import pandas as pd

SHORTEST_SENTENCES = True


def test_train_dataset(train_size=100, test_size=20, random_state=None):
    df = pd.read_csv(join(dirname(__file__), f"../corpus/IMDB_preprocessed.csv"))
    per_class_chunk_size = train_size + test_size

    # select shortest sentences in the dataset
    if SHORTEST_SENTENCES:
        df = df.sort_values(by="review", key=lambda review: review.apply(len))
        train_test_positive_class = df[df["sentiment"] == 1].iloc[:per_class_chunk_size]
        train_test_negative_class = df[df["sentiment"] == 0].iloc[:per_class_chunk_size]
    else:
        params = {
            "n": per_class_chunk_size,
            "replace": False,
            "random_state": random_state,
        }
        train_test_positive_class = df[df["sentiment"] == 1].sample(**params)
        train_test_negative_class = df[df["sentiment"] == 0].sample(**params)

    train_set = [
        train_test_positive_class.iloc[:train_size],
        train_test_negative_class.iloc[:train_size],
    ]

    test_set = [
        train_test_positive_class.iloc[train_size:],
        train_test_negative_class.iloc[train_size:],
    ]

    train_set = pd.concat(train_set).sample(frac=1)
    test_set = pd.concat(test_set).sample(frac=1)

    print("=== train set ===")
    print(train_set["sentiment"].value_counts())
    print("=== test set ===")
    print(test_set["sentiment"].value_counts())

    return train_set, test_set


if __name__ == "__main__":
    train, test = test_train_dataset(train_size=50, test_size=10)
    print(train.shape, test.shape)
