import numpy as np
from matplotlib import pyplot as plt

from PymoNNto import Behaviour, SynapseGroup, Recorder, NeuronGroup, Network
from PymoNNto.Exploration.Network_UI import get_default_UI_modules, Network_UI
from src.libs import behaviours
from src.libs.data_generator_numpy import stream_generator_for_character
from src.libs.environment import set_dopamine, get_dopamine
from src.libs.helper import behaviour_generator


# ================= BEHAVIOURS  =================


class NeuronBehaviour(Behaviour):
    def set_variables(self, neurons: NeuronGroup):
        print("NeuronBehaviour", neurons)

    def new_iteration(self, neurons):
        print("NeuronBehaviour 🆕", neurons.tags[0], neurons.iteration)


class SynapseBehaviour(Behaviour):
    def set_variables(self, synapse: SynapseGroup):
        synapse.enabled = self.get_init_attr("enabled", True, synapse)
        synapse.delay = synapse.get_synapse_mat("uniform")
        synapse.weights_scale = synapse.get_synapse_mat_dim()
        print("SynapseBehaviour", synapse)

    def new_iteration(self, synapse):
        print("SynapseBehaviour 🆕", synapse.iteration)


# ================= NETWORK  =================
def main():
    network = Network()
    neurons_configs = {
        "net": network,
        "size": 1,
        "behaviour": behaviour_generator([NeuronBehaviour()]),
    }
    neuron_i = NeuronGroup(tag="i", **neurons_configs)
    neuron_j = NeuronGroup(tag="j", **neurons_configs)

    SynapseGroup(
        net=network,
        src=neuron_i,
        dst=neuron_j,
        tag="GLUTAMATE",
        behaviour=behaviour_generator([SynapseBehaviour()]),
    )

    network.initialize()
    network.simulate_iterations(3, measure_block_time=True)


if __name__ == "__main__":
    main()
