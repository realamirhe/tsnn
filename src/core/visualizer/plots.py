# ================= Visualization =================
from matplotlib import pyplot as plt


def voltage_visualizer(voltage, title="voltage_trace", labels=("iteration", "voltage")):
    plt.plot(voltage)
    plt.title(title)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.show()


def spike_visualizer(spikes, title="spike_trace", labels=("iteration", "spike")):
    if not spikes:
        return
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
            network[f"{ng}-recorder", 0]["n.v", 0],
            title=f"{ng} spike activity",
        )
