import random

import numpy as np

# ================= RESET RANDOM SEED =================
from matplotlib import pyplot as plt


def reset_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


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
