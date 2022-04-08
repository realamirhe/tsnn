import unittest

from PymoNNto import Network, NeuronGroup, Recorder
from src.core.neurons.neurons import StreamableLIFNeurons
from src.data.constants import letters
from src.data.spike_generator import spike_stream_i
from src.helpers.base import behaviour_generator

V_RESET = -65.0


def make_custom_network(corpus):
    network = Network()

    lif_base = {
        "v_rest": -67.0,
        "v_reset": V_RESET,
        "threshold": -52,
        "dt": 1.0,
        "R": 2,
        "tau": 3,
        "stream": [spike_stream_i(char) for char in corpus],
    }

    NeuronGroup(
        net=network,
        tag="letters",
        size=len(letters),
        behaviour=behaviour_generator(
            [
                StreamableLIFNeurons(**lif_base),
                Recorder(tag="recorder", variables=["n.v", "n.fired"]),
            ]
        ),
    )
    network.initialize()
    network.simulate_iterations(len(corpus))
    return network


class MyTestCase(unittest.TestCase):
    def test_network_force_fire_mechanism(self):
        corpus = "abcabab"
        net = make_custom_network(corpus)
        self.assertEqual(net.iteration, len(corpus))
        ng_fired = net["recorder", 0]["n.fired", 0]
        for fire_pattern, letter in zip(ng_fired, corpus):
            index_letters = letters.index(letter)
            for inx in range(len(letters)):
                if inx == index_letters:
                    self.assertTrue(fire_pattern[inx])
                else:
                    self.assertFalse(fire_pattern[inx])

    def test_network_should_reset_voltage_on_spike(self):
        corpus = "abc ab a b"
        net = make_custom_network(corpus)
        ng_fired = net["recorder", 0]["n.fired", 0]
        ng_v = net["recorder", 0]["n.v", 0]

        for v, fired in zip(ng_v, ng_fired):
            if fired.any():
                self.assertEqual(v[fired], V_RESET)


if __name__ == "__main__":
    unittest.main()
