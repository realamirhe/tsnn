import string

import matplotlib.pyplot as plt
import nltk
import torch

from bindsnet.analysis.plotting import plot_spikes, plot_voltages
from bindsnet.learning import PostPre, MSTDP, MSTDPET
from bindsnet.network import Network
from bindsnet.learning.reward import MovingAvgRPE
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection

# RESET RANDOM SEED
seed = 42
torch.manual_seed(seed)
# random.seed(seed)
# np.random.seed(seed)

# codes
letters = string.ascii_lowercase

# TODO: delay
# TODO: LRNU
# TODO: LIFNodes decay
# TODO: adaptive neurons, with high adaptation rate which cause to eliminate

"""
two words with non overlapping characters
random noise permutation with exact match in the middle
lr ~ e1-4 => spiking epoch training phase
delay for word scramble
decay fast but delay affects more!

setting spike to zero
set voltage just a little bit than thresh hold
Force Spike
Keep traces so mess thing up
inject_v
"""

# No Long term learning, but active process make active spikes easily
# see see see see see see see see see
corpus = """
ssssssssssssssssssssssssssssssssssssssssssssss
cccccccccccccccccccccccccc
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

# esh is the same as she
words = ["she", "cat", "swim", "day", "see", "not"]
# # words = ["she"]
tokens = nltk.word_tokenize(corpus)
sentences_token = [nltk.word_tokenize(sent) for sent in corpus.split(".")]


def char2spike(char):
    spike = torch.zeros(len(letters))

    if char == " ":
        return spike
    if char == ".":
        return spike
        # return -torch.ones_like(spike) # NOTE: because of converting to byte it will be 255

    spike[letters.index(char)] = 1  # TODO: change weight connections
    # spike[letters.index(char)] = 50  # no effect on spikes but make voltage decay more smooth
    return spike


def words_letter_weights(words_list):
    weights = []
    for word in words_list:
        # NOTE: large inhibition interrupt learning phase / adding random variant 🌟 will not work in this case too.
        # stimulus = -torch.ones(len(letters))
        # 0.6 is the scaling size make the weights -0.1 for the inhibition
        stimulus = -1 * torch.ones(len(letters))
        stimulus += 0.1 * torch.rand_like(stimulus)
        word_size = len(word)
        for char_idx, char in enumerate(word):
            # NOTE: fix weight makes the voltage decay more aggressive
            # stimulus[letters.index(char)] = 1  # fixed weights 🌟
            stimulus[letters.index(char)] = (word_size - char_idx) / word_size
        weights.append(stimulus)
    return torch.stack(weights)


if __name__ == "__main__":
    # NOTE: Increasing repeat count only effects input neurons not words spikes count
    repeat = 10
    joined_tokens = " ".join(tokens * repeat)
    time = len(joined_tokens)

    network = Network(reward_fn=MovingAvgRPE)  # batch_size=10 have no effect

    letters_layer = Input(n=len(letters), traces=True)
    network.add_layer(layer=letters_layer, name="letters")

    # Also tested with words * 2 [not any progress], Note: comment w and plot ticks part
    words_layer = LIFNodes(n=len(words), traces=True)
    network.add_layer(layer=words_layer, name="words")

    # Create connection between input and output layers.
    forward_connection = Connection(
        source=letters_layer,
        target=words_layer,
        # NOTE: increasing w bound will decrease the firing rate (maybe be more space to search lost in the network)
        wmin=-3,  # minimum weight value
        wmax=3,  # maximum weight value
        # update_rule=MSTDPET,  # learning rule + <eligibility trace/>
        # nu=1e-1,  # learning rate
        # norm=0.5 * letters_layer.n,  # normalization (NOTE: doesn't effect so much with MSTDPET base config)
        # update_rule=MSTDP,
        # TODO: test without w
        update_rule=PostPre,
        # nu=(1e-4, 1e-2),  # 🌟 bounding learning rate makes the learning much more difficult
        # Normal(0.05, 0.01) weights.
        w=0.05 + 0.1 * torch.randn(letters_layer.n, words_layer.n),
        # w is in [-0.05, 0.15]
        # w=0.05 + 0.1 * torch.transpose(words_letter_weights(words), 0, 1),
    )
    network.add_connection(forward_connection, source="letters", target="words")

    # 🌚 new section
    recurrent_connection = Connection(
        source=words_layer,
        target=words_layer,
        update_rule=PostPre,
        # Small, inhibitory "competitive" weights.
        # w=0.025 * (torch.eye(words_layer.n) - 1),
    )
    network.add_connection(recurrent_connection, source="words", target="words")

    letters_monitor = Monitor(obj=letters_layer, state_vars=("s",), time=time)
    words_monitor = Monitor(obj=words_layer, state_vars=("s", "v"), time=time)

    network.add_monitor(monitor=letters_monitor, name="letters")
    network.add_monitor(monitor=words_monitor, name="words")

    input_data = [char2spike(char) for char in joined_tokens]
    input_data = torch.stack(input_data).byte()
    inputs = {"letters": input_data}

    # print(visualization.summary(network))
    # Multi epochs (epochs=3) makes the neurons not to spike any more!
    network.run(
        inputs=inputs,
        time=time,
        # increasing reward scalar makes
        # the more reward the more neurons will spike (in MSTDPET not in PrePost)
        reward=100000.0,
    )

    spikes = {
        "letters": letters_monitor.get("s"),
        "words": words_monitor.get("s"),
    }  # words monitor returns Tensor<bool>
    voltages = {"words": words_monitor.get("v")}

    plt.ioff()
    img, axes = plot_spikes(spikes)
    axes[1].set_yticks([i for i in range(len(words))])
    axes[1].set_yticklabels(words)
    plot_voltages(voltages, plot_type="line")
    # plot_voltages(voltages, plot_type="color")
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
-------------------------------------------------------------------------------------------
- [ ] defer for batching letters

[input stimulus for letters]
- [ ] change input data to spike letters with a small window in parallel e.g. car is going to activate all c, a, r at once
- [x] letter by letter sequential neuron activation

[forward connection] 
- [x] adding random variants to wrights 🌟

[Appendix]
🌟 seems to reduce the neuron firing rate 
"""
