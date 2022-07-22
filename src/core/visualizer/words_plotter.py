from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS

from src.helpers.network import EpisodeTracker

stopwords = set(STOPWORDS)


def wordcloud_plotter(title, data: str):
    wordcloud = WordCloud(
        width=800,
        height=660,
        background_color="white",
        stopwords=stopwords,
        min_font_size=10,
    ).generate_from_text(data)
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud)
    plt.title(title + f" (eps={EpisodeTracker.episode()})", fontsize=13)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()


if __name__ == "__main__":
    wordcloud_plotter("pos", "long stage new talent")
