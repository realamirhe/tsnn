import random

import numpy as np

# ================= RESET RANDOM SEED =================
from matplotlib import pyplot as plt


def reset_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


# ================= Array manipulator =================
def re_range_binary(array):
    return np.where(array > 0, 1, -1)


# ================= Behaviours Maker =================
def behaviour_generator(behaviours):
    return {index + 1: behaviour for index, behaviour in enumerate(behaviours)}


# ================= Visualization =================
def voltage_visualizer(voltage, title="voltage_trace", labels=("iteration", "voltage")):
    plt.plot(voltage)
    plt.title(title)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.show()


def spike_visualizer(spikes, title="spike_trace", labels=("iteration", "spike")):
    plt.imshow(spikes, cmap="gray", aspect="auto")
    plt.title(title)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.show()


def raster_plots(network, ngs=["letters", "words"]):
    for ng in ngs:
        spike_visualizer(
            network[f"{ng}-recorder", 0]["n.fired", 0, "np"].transpose(),
            title=f"{ng} spike activity",
        )


def voltage_plots(network, ngs=["letters", "words"]):
    for ng in ngs:
        voltage_visualizer(
            network[f"{ng}-recorder", 0]["n.v", 0], title=f"{ng} spike activity",
        )
