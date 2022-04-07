import numpy as np
from matplotlib import pyplot as plt

from PymoNNto import Behaviour, SynapseGroup, Recorder, NeuronGroup, Network
from src.trunk.libs.helper import behaviour_generator, reset_random_seed

reset_random_seed(41)


# ================= BEHAVIOURS  =================
class LIFNeuron(Behaviour):
    def set_variables(self, neurons):
        self.set_init_attrs_as_variables(neurons)
        neurons.v = neurons.get_neuron_vec() * neurons.v_rest
        neurons.spikes = neurons.get_neuron_vec() > neurons.threshold
        neurons.dt = 1.0

    def new_iteration(self, neurons):
        neurons.I = 20 * neurons.stream[neurons.iteration - 1]
        dv_dt = (neurons.v_rest - neurons.v) + neurons.R * neurons.I
        neurons.v += dv_dt * neurons.dt / neurons.tau
        neurons.fired = neurons.v >= neurons.threshold
        if np.sum(neurons.fired) > 0:
            neurons.v[neurons.fired] = neurons.v_reset


class PairWiseSTDP(Behaviour):
    def set_variables(self, synapses):
        self.set_init_attrs_as_variables(synapses)
        synapses.W = synapses.get_synapse_mat("uniform")
        synapses.src.trace = synapses.src.get_neuron_vec()
        synapses.dst.trace = synapses.dst.get_neuron_vec()

    def new_iteration(self, synapses):
        dx = -synapses.src.trace / synapses.tau_plus + synapses.src.fired
        dy = -synapses.dst.trace / synapses.tau_minus + synapses.dst.fired
        synapses.src.trace += dx * synapses.dt
        synapses.dst.trace += dy * synapses.dt

        dw_minus = -synapses.a_minus * synapses.dst.trace * synapses.src.fired
        dw_plus = synapses.a_plus * synapses.src.trace * synapses.dst.fired

        synapses.W = np.clip(
            synapses.W + (dw_plus + dw_minus) * synapses.dt,
            synapses.w_min,
            synapses.w_max,
        )


# ================= SETUP  =================
size = 20
i_stream = np.random.randint(0, 2, size)
j_stream = np.random.randint(0, 2, size)

xj_stream = np.sin(j_stream)
yi_stream = np.cos(i_stream)
wij_stream = xj_stream + yi_stream


# ================= NETWORK  =================
def main():
    network = Network()
    lif_base = {
        "v_rest": -65,
        "v_reset": -65,
        "threshold": -52,
        "dt": 1.0,
        "R": 2,
        "tau": 3,
    }

    i_ng = NeuronGroup(
        net=network,
        tag="i_ng",
        size=1,
        behaviour=behaviour_generator(
            [
                LIFNeuron(tag="lif:train", stream=j_stream, **lif_base),
                Recorder(tag="i-recorder", variables=["n.v", "n.fired"]),
            ]
        ),
    )

    j_ng = NeuronGroup(
        net=network,
        tag="j_ng",
        size=1,
        behaviour=behaviour_generator(
            [
                LIFNeuron(tag="lif:train", stream=i_stream, **lif_base),
                Recorder(tag="j-recorder", variables=["n.v", "n.fired"]),
            ]
        ),
    )

    synapse_base = {
        "tau_plus": 3,
        "tau_minus": 3,
        "a_plus": 2.15,
        "a_minus": 2.15,
        "w_max": 10,
        "w_min": 0,
        "dt": 1.0,
    }
    SynapseGroup(
        net=network,
        src=i_ng,
        dst=j_ng,
        tag="GLUTAMATE",
        behaviour=behaviour_generator(
            [
                PairWiseSTDP(**synapse_base),
                Recorder(
                    tag="s-recorder", variables=["s.src.trace", "s.dst.trace", "s.W"]
                ),
            ]
        ),
    )

    network.initialize()
    network.simulate_iterations(size, measure_block_time=True)

    fig, axes = plt.subplots(5)
    for ax in axes:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

    (xj, j, i, yi, wij) = axes

    xj.plot(network[f"s-recorder", 0]["s.src.trace", 0])
    xj.set_ylabel("xj", rotation=45)

    i.stem(i_stream, linefmt="grey", markerfmt=".")
    i.set_ylabel("i", rotation=45)

    j.stem(j_stream, linefmt="grey", markerfmt=".")
    j.set_ylabel("j", rotation=45)

    yi.plot(network[f"s-recorder", 0]["s.dst.trace", 0])
    yi.set_ylabel("yi", rotation=45)

    w = [i[0][0] for i in network[f"s-recorder", 0]["s.W", 0]]
    wij.plot(w)
    wij.set_ylabel("wij", rotation=45)

    plt.show()


if __name__ == "__main__":
    main()
