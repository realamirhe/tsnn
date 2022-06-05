import unittest

import numpy as np

from PymoNNto import Network, NeuronGroup, Recorder
from src import configs
from src.core.neurons.neurons import StreamableLIFNeurons
from src.data.spike_generator import spike_stream_i
from src.helpers.base import behaviour_generator

LIF_BASE = {
    "v_rest": -67.0,
    "v_reset": -65.0,
    "threshold": -52,
    "dt": 1.0,
    "R": 2,
    "tau": 3,
}


def make_custom_network(corpus, use_stream=True, use_long_term_effect=True):
    network = Network()

    NeuronGroup(
        net=network,
        tag="letters",
        size=len(configs.corpus.letters),
        behaviour=behaviour_generator(
            [
                StreamableLIFNeurons(
                    **LIF_BASE,
                    has_long_term_effect=use_long_term_effect,
                    stream=[spike_stream_i(char) for char in corpus]
                    if use_stream
                    else None,
                ),
                Recorder(tag="recorder", variables=["n.v", "n.fired"]),
            ]
        ),
    )
    network.initialize()
    network.simulate_iterations(len(corpus))
    return network


class NeuronGroupTestCase(unittest.TestCase):
    def test_network_force_fire_mechanism(self):
        corpus = "abcabab"
        net = make_custom_network(corpus)
        ng_fired = net["recorder", 0]["n.fired", 0]
        for fire_pattern, letter in zip(ng_fired, corpus):
            index_letters = configs.corpus.letters.index(letter)
            for inx in range(len(configs.corpus.letters)):
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
                self.assertEqual(v[fired], LIF_BASE["v_reset"])

    def test_network_should_use_default_configs(self):
        corpus = "abc ab a b"
        net = make_custom_network(corpus, use_long_term_effect=False)
        ng = net.NeuronGroups[0]
        for key, value in LIF_BASE.items():
            if key == "dt":
                continue
            self.assertEqual(getattr(ng, key), value)

        net = make_custom_network(corpus)
        ng = net.NeuronGroups[0]
        for key, value in LIF_BASE.items():
            if key == "dt":
                continue
            ng_attr = getattr(ng, key)
            if key == "threshold":
                self.assertTrue(np.all(ng_attr == ng_attr[0]))
                self.assertEqual(ng_attr[0], value)
            else:
                self.assertEqual(ng_attr, value)

    # TODO: should neuron update itself when no stimulus is applied?
    # def test_network_must_keeps_voltage_on_no_stimulus(self):
    #     corpus = "abc ab a b"
    #     net = make_custom_network(corpus, use_stream=False)
    #     ng_v = net["recorder", 0]["n.v", 0]
    #     ng_v0 = ng_v[0]
    #     for vs in ng_v:
    #         for v, v0 in zip(vs, ng_v0):
    #             self.assertEqual(v, v0)


if __name__ == "__main__":
    unittest.main()
