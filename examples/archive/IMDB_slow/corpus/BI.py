import numpy as np
import pandas as pd


def report_info(f):
    def wrapper():
        print(f"=== {f.__name__} ===")
        f()
        print()

    return wrapper


@report_info
def review_words_count():
    df = pd.read_csv("IMDB_preprocessed.csv")
    df["review"] = df["review"].apply(lambda x: len(x.split(" ")))
    print(df["review"].describe())


@report_info
def review_words_length():
    df = pd.read_csv("IMDB_preprocessed.csv")
    length = []
    for idx, row in df.iterrows():
        length.extend(list(map(len, row["review"].split(" "))))
        if np.max(length[-120:]) > 20:
            print(list(filter(lambda x: len(x) >= 20, row["review"].split(" "))))

    length = np.array(length)
    length = length[length <= 70]
    print(f"max={np.max(length)} min={np.min(length)}")
    length = pd.Series(length)
    print(length.head())
    print(length.describe())


if __name__ == "__main__":
    review_words_count()
    review_words_length()

# W d  [word] # preprocessing (w, d)
# nlp interation,
# popluation -> classification
