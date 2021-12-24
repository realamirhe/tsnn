import string

import matplotlib.pyplot as plt
import nltk
import torch

from bindsnet.analysis.plotting import plot_spikes, plot_voltages
from bindsnet.learning import PostPre
from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, LIFNodes

# codes
from bindsnet.network.topology import Connection

letters = string.ascii_lowercase

corpus = """
She does not study German on Monday.
Does she live in Paris.
He does not teach math.
Cats hate water.
Every child likes an ice cream.
My brother takes out the trash.
The course starts next Sunday.
She swims every morning.
I do not wash the dishes.
We see them every week.
""".lower()

words = ["she", "cat", "swim", "day", "see", "not"]
tokens = nltk.word_tokenize(corpus)
sentences_token = [nltk.word_tokenize(sent) for sent in corpus.split(".")]


def char2spike(char):
    spike = torch.zeros(len(letters))

    if char == " ":
        return spike
    if char == ".":
        return spike
        # return -torch.ones_like(spike) # NOTE: because of converting to byte it will be 255

    spike[letters.index(char)] = 255
    return spike


def words_letter_weights(words_list):
    weights = []
    for word in words_list:
        stimulus = -torch.ones(len(letters))
        # stimulus += 0.1 * torch.rand_like(stimulus)  # random variant 🌟
        word_size = len(word)
        for char_idx, char in enumerate(word):
            # stimulus[letters.index(char)] = 1  # fixed weights 🌟
            stimulus[letters.index(char)] = (word_size - char_idx) / word_size
        weights.append(stimulus)
    return torch.stack(weights)


if __name__ == "__main__":
    joined_tokens = " ".join(tokens)
    time = len(joined_tokens)

    network = Network()

    letters_layer = Input(n=len(letters), traces=True)
    network.add_layer(layer=letters_layer, name="letters")

    words_layer = LIFNodes(n=len(words), traces=True)
    network.add_layer(layer=words_layer, name="words")

    # Create connection between input and output layers.
    forward_connection = Connection(
        source=letters_layer,
        target=words_layer,
        update_rule=PostPre,
        nu=(1e-4, 1e-2),
        # Normal(0.05, 0.01) weights.
        # w=0.05 + 0.1 * torch.randn(letters_layer.n, words_layer.n),
        w=torch.transpose(words_letter_weights(words), 0, 1),
    )
    network.add_connection(forward_connection, source="letters", target="words")

    letters_monitor = Monitor(obj=letters_layer, state_vars=("s",), time=time)
    words_monitor = Monitor(obj=words_layer, state_vars=("s", "v"), time=time)

    network.add_monitor(monitor=letters_monitor, name="letters")
    network.add_monitor(monitor=words_monitor, name="words")

    input_data = [char2spike(char) for char in joined_tokens]
    input_data = torch.stack(input_data).byte()
    inputs = {"letters": input_data}

    network.run(inputs=inputs, time=time)

    spikes = {"letters": letters_monitor.get("s"), "words": words_monitor.get("s")}
    voltages = {"words": words_monitor.get("v")}

    plt.ioff()
    plot_spikes(spikes)
    plot_voltages(voltages, plot_type="line")
    plt.show()

    # print(input_data.shape)
    # print(input_data)
    # target_layer = LIFNodes(n=1000, traces=True)

    # print(tokens)
    # print(sentences_token)
    # print(list(set(tokens)))
    # print(char2spike('a'))
    # print(char2spike('b'))
    # print(char2spike('c'))
    # print(char2spike('z'))

    # print(words_letter_weights(words))

"""
Must be checked
[connection from letters to words]
- [x] existing === excitation adn visa versa 🌟
- [x] position matters first letter excite the most and the last letter spikes the least
- [ ] attention based weights for connections, first and last characters are more important

[input stimulus for letters]
- [ ] change input data to spike letters with a small window in parallel e.g. car is going to activate all c, a, r at once
- [x] letter by letter sequential neuron activation

[forward connection] 
- [x] adding random variants to wrights 🌟

[Appendix]
🌟 seems to reduce the neuron firing rate 
"""
