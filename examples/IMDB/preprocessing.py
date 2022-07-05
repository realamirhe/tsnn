import contractions
import nltk
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.preprocessing import LabelEncoder

try:
    stop_words = stopwords.words("english")
except:
    nltk.download("stopwords")
    stop_words = stopwords.words("english")

try:
    word_net_lemmatizer = WordNetLemmatizer()
    word_net_lemmatizer.lemmatize("better")
except:
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    word_net_lemmatizer = WordNetLemmatizer()

punctuation_remover_tokenizer = RegexpTokenizer(r"\w+")


def pre_process_imdb_dataset():
    df = pd.read_csv("./IMDB Dataset.csv")
    # positive=1 & negative=0
    le = LabelEncoder()
    df["sentiment"] = le.fit_transform(df["sentiment"])
    # process reviews
    df["review"] = df["review"].apply(pre_process_text)

    return df


def pre_process_text(txt: str) -> str:
    txt = txt.lower()
    # Remove HTML Tags
    txt = BeautifulSoup(txt, "html.parser").text
    # Fixing contractions (e.g. gonna => going to)
    txt = contractions.fix(txt)
    # Remove punctuation (e.g. $, #) & stop words (e.g. is, there, he's)
    txt = [
        word
        for word in punctuation_remover_tokenizer.tokenize(txt)
        if word not in stop_words
    ]
    # Lemmatizing (e.g. better => good)
    txt = [word_net_lemmatizer.lemmatize(word) for word in txt]
    # Stemming (e.g. universally => univers, single => singl)
    # porter = PorterStemmer()
    # text = [porter.stem(word) for word in text]
    # NOTE: stemming may corrupt the unique noun words e.g. like `amirhe`
    return " ".join(txt)


if __name__ == "__main__":
    data = pre_process_imdb_dataset()
    print(data.head())
    print(data.sentiment.value_counts())
    data.to_csv("./IMDB_preprocessed.csv", index_label=False, index=False)
