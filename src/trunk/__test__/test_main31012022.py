import unittest

import numpy as np

from src.trunk.main31012022 import Delay


class Neurons:
    def __init__(self, size):
        self.fired = np.random.randint(0, 2, size)
        self.size = size
        self.max_delay = 3

    def get_init_attr(self, key, default, n):
        return getattr(self, key, default)


class TestDelay(unittest.TestCase):
    def test_something(self):
        instance = Delay()
        n = Neurons(5)
        instance.set_variables(n)

        instance.new_iteration(n)
        instance.new_iteration(n)
        self.assertFalse(np.all(instance.delayed_spikes == 0))
        instance.new_iteration(n)

        self.assertTrue(np.all(instance.delayed_spikes == 0))


if __name__ == "__main__":
    unittest.main()
