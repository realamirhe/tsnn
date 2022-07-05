import numpy as np
import pandas as pd
from nltk.probability import FreqDist


def extract_common_imdb_review_words(size: int) -> np.ndarray:
    df = pd.read_csv("../corpus/IMDB_preprocessed.csv")
    freq_dist = FreqDist()
    for review in df["review"]:
        for word in review.split(" "):
            freq_dist[word] += 1

    common_words = freq_dist.most_common(size)
    common_words = [key for (key, _freq) in common_words]

    return np.array(common_words)


if __name__ == "__main__":
    # we have 94365 unique words, so we get 80% for them
    # 15540 gte 25, 10210 gte 50
    common_words = extract_common_imdb_review_words(20_000)
    with open("../corpus/common_words.npy", "wb") as file:
        np.save(file, common_words)
    with open("../corpus/common_words.npy", "rb") as file:
        print(np.load(file))
