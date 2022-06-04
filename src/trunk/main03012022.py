"""
i -> w -> j

d = 3
buffer = [0, 0, 0]
l0_spike_i => buffer = [1, 0, 0]
l2_non_spike_i => buffer = [0, 1, 0] => 0 out put of neuron
l3_non_spike_i => buffer = [0, 0, 1] => 0 out put of neuron
l4_spike_i => buffer = [1, 0, 0] => 1 out put of neuron
"""
# fmt: off
import random
import string
from typing import List, Union

import matplotlib.pyplot as plt
import torch
from bindsnet.analysis.plotting import plot_spikes, plot_voltages
from bindsnet.learning import PostPre
from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, AdaptiveLIFNodes
from bindsnet.network.topology import Connection

"""
LMU ✔
two words with non overlapping characters ✔
random noise permutation with exact match in the middle ✔
Force Spike ✔

delay for word scramble 
    - nengo ✔
    - lmu source code ✔
    - delay mechanism for input connections ✔
adaptive neurons, with high adaptation rate which cause to eliminate ✔
decay fast + more effective delay 👋 

# [TECHNICAL STUFF]
setting spike to zero
set voltage just a little bit than thresh hold
lr ~ e1-4 => spiking epoch training phase
inject_v
Keep traces so it doesn't mess things up
"""

# RESET RANDOM SEED
seed = 42
torch.manual_seed(seed)
random.seed(seed)
# np.random.seed(seed)

# SETUP
letters = string.ascii_lowercase
# cat, foo, pop (same char), bike (4-letter)
words = ["abc", "omn"]
words = [word.lower() for word in words]
assert (
        len(set(words[0]).intersection(words[1])) == 0
), "words must be non overlapping for first stage"


def gen_corpus(
        size=100, prob=0.5, min_length=1, max_length=10, no_common_chars: bool = False
) -> List[str]:
    """
        Generate a corpus of random words. contains learnable words within
    """
    corpus: List[str] = []
    valid_letters = letters
    if no_common_chars:
        words_character = set("".join(words))
        valid_letters = set(valid_letters).symmetric_difference(words_character)
        valid_letters = "".join(valid_letters)

    print(f"valid letters: {valid_letters}")
    for i in range(size):
        if random.random() < prob:
            word = random.choice(words)
        else:
            word_length = random.randint(min_length, max_length)
            word = "".join(random.choice(valid_letters) for _ in range(word_length))
        corpus.append(word)
    return corpus


def char2spike(char: str) -> torch.Tensor:
    """
        Convert a character to a spike vector
        :param char: character to convert
        :return: spike vector
    """
    spike = torch.zeros(len(letters))
    if char != " ":
        spike[letters.index(char)] = 1
    return spike


def words_letter_weights(words: List[str]) -> torch.Tensor:
    """
        Generate a weight matrix for the network from letters to words
        :return weight matrix tensor from letters to words
    """
    positive_connection = 5

    weights = torch.rand((len(letters), len(words))) - 0.5
    for word_idx, word in enumerate(words):
        for char in word:
            weights[letters.index(char), word_idx] = positive_connection

    return weights


def words_words_weights(words) -> torch.Tensor:
    """
        Generate a weight matrix for the network from words to words
        :return weight matrix tensor from words to words
    """
    negative_connection = -5
    weights = torch.zeros((len(words), len(words)))
    for word_idx, word in enumerate(words):
        weights[word_idx, :] = negative_connection
        weights[word_idx, word_idx] = 0
    return weights


def simulated_accumulated_delay(
        input_data: torch.Tensor,
        delay: int,
        target_strategy: Union["sequence", "words"] = "sequence",
) -> torch.Tensor:
    assert delay >= 0, "delay must be non-negative"
    if delay == 0:
        print("delay must be greater than 0, 0 delay has no impact on the output")
        return input_data
    # asakjdfh abc
    # asakjdf h a ha hab a ab abc
    # home abc
    # [a, ab, abc] -> conncetion nueron abc 👋
    #

    if target_strategy == "sequence":
        for neuron_idx in range(input_data.size(0) - 1, -1, -1):
            if neuron_idx < delay:
                continue
            input_data[neuron_idx, :] += (
                input_data[neuron_idx - delay: neuron_idx, :]
                    .sum(dim=0)
                    .clamp(min=0, max=1)
            )

    # it is redundant for sequence
    return input_data.byte()


if __name__ == "__main__":
    corpus = gen_corpus(size=30, prob=0.5, no_common_chars=False)
    # corpus = ["abc"] * 5
    corpus = " ".join(corpus)

    # simulation time in dt count version
    # TODO: convert it to seconds
    # TODO: use the size parameter to define time
    time = len(corpus)

    network = Network()

    # Letters layer and monitors
    letters_layer = Input(
        n=len(letters),
        traces=True,
        # tc_trace=20.0, # spike trace decay time.
        # trace_scale=0.001, # scaling factor
        sum_input=True,
    )
    network.add_layer(layer=letters_layer, name="letters")
    letters_monitor = Monitor(letters_layer, state_vars=("s",), time=time)
    network.add_monitor(monitor=letters_monitor, name="letters")

    # Words layer and monitors
    # words_layer = LIFNodes(
    words_layer = AdaptiveLIFNodes(
        n=len(words),
        traces=True,
        tc_trace=20.0,  # [200] spike trace decay
        trace_scale=1.0,
        sum_input=False,
        rest=-65.0,
        reset=-65.0,
        thresh=-52.0,
        refrac=0,  # [5] 👽 non-biological view
        tc_decay=5.0,  # 🌟 [100.0] reduce
        theta_plus=0.05,  # 🌟 should be increased?
        tc_theta_decay=1e7,
        lbound=-70,  # Prevent so much decay when no input is received
        # traces_additive=True,
        # sum_input=True
    )
network.add_layer(layer=words_layer, name="words")
words_monitor = Monitor(words_layer, state_vars=("s", "v"), time=time)
network.add_monitor(monitor=words_monitor, name="words")

# connections [feed-forward]
forward_connection = Connection(
    # forward_connection = DelayedConnection(
    source=letters_layer,
    target=words_layer,
    # learning
    update_rule=PostPre,
    # weight_decay=10,
    norm=1.0,
    # w=0.05 + 0.1 * torch.randn(letters_layer.n, words_layer.n),
    w=words_letter_weights(words),
    # delayed arguments
    # dmax=len(letters),
    # d=torch.randint(low=0, high=len(words), size=(len(letters), 1)),
)
# FIXME: HACK
words_layer.d = len(letters)
network.add_connection(connection=forward_connection, source="letters", target="words")

recurrent_connection = Connection(
    source=words_layer,
    target=words_layer,
    # Noop learning
    w=words_words_weights(words),
)
network.add_connection(connection=recurrent_connection, source="words", target="words")

# convert characters to spikes tensor
input_data = [char2spike(char) for char in corpus]
input_data = torch.stack(input_data).byte()
# delay can be e.g. max len words
input_data = simulated_accumulated_delay(
    input_data, delay=3, target_strategy="sequence"
)

inputs = {"letters": input_data}
network.run(inputs=inputs, time=time, progress_bar=True)

spikes = {"letters": letters_monitor.get("s"), "words": words_monitor.get("s")}
voltages = {"words": words_monitor.get("v")}

plt.ioff()
img, axes = plot_spikes(spikes)
axes[1].set_yticks([i for i in range(len(words))])
axes[1].set_yticklabels(words)
plot_voltages(voltages, plot_type="line")
plt.show()

"""
Report:
base config: 
- no learning seems to be happen with 200, 3000 corpus size
"""

# fmt: on
