import unittest

import numpy as np

from PymoNNto import Network, NeuronGroup, Recorder
from src.configs import corpus_config
from src.core.neurons.neurons import StreamableLIFNeurons
from src.core.stabilizer.winner_take_all import WinnerTakeAll
from src.helpers.base import behaviour_generator
from src.safeguards.libs.OverrideNeurons import OverrideNeurons, OVERRIDABLE_SUFFIX


def make_custom_network(size, overridable_params=None):
    if not np.all(
        [key.endswith(OVERRIDABLE_SUFFIX) for key in overridable_params.keys()]
    ):
        raise AssertionError(f"All key must ends with {OVERRIDABLE_SUFFIX}")

    network = Network()
    NeuronGroup(
        net=network,
        tag="letters",
        size=len(corpus_config.words),
        behaviour=behaviour_generator(
            [
                StreamableLIFNeurons(),
                OverrideNeurons(**overridable_params),
                WinnerTakeAll(),
                Recorder(tag="recorder", variables=["n.v", "n.fired"]),
            ]
        ),
    )
    network.initialize()
    network.simulate_iterations(size)
    return network


class WinnerTakeAllTestCase(unittest.TestCase):
    def test_fire_pattern_must_no_change_randomly(self):
        vector_size = len(corpus_config.words)
        params = {
            f"v_{OVERRIDABLE_SUFFIX}": np.ones(vector_size) * -70,
            f"fired_{OVERRIDABLE_SUFFIX}": np.zeros(vector_size) > 0,
        }
        net = make_custom_network(1, params)
        ng_fired = net["recorder", 0]["n.fired", 0]
        self.assertTrue(np.all(ng_fired == params[f"fired_{OVERRIDABLE_SUFFIX}"]))

    def test_fire_pattern_must_no_change_randomly(self):
        vector_size = len(corpus_config.words)
        times = 30
        params = {
            f"v_{OVERRIDABLE_SUFFIX}": np.random.random((times, vector_size)),
            f"fired_{OVERRIDABLE_SUFFIX}": np.random.random((times, vector_size)) > 0.5,
        }
        net = make_custom_network(1, params)
        ng_fired = net["recorder", 0]["n.fired", 0]
        # at most one neuron should fire
        for fired in ng_fired:
            self.assertLessEqual(np.sum(fired), 1)


if __name__ == "__main__":
    unittest.main()
