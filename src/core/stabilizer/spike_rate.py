from collections import Counter

import numpy as np

from PymoNNto import Behaviour


class SpikeRate(Behaviour):
    __slots__ = ["interval_size", "outputs"]

    def set_variables(self, n):
        self.interval_size = self.get_init_attr("interval_size", 5, n)
        self.outputs = self.get_init_attr("outputs", [], n)
        self.prediction_history = []

    def new_iteration(self, n):
        self.prediction_history.append(n.fired)
        if len(self.prediction_history) == self.interval_size:
            output_history = self.outputs[-self.interval_size :]  # [True, False],
            bit_range = 1 << np.arange(self.outputs[0].size)

            outputs = Counter(
                [o.dot(bit_range) for o in output_history if not np.isnan(o).any()]
            )
            predictions = Counter([p.dot(bit_range) for p in self.prediction_history])

            for neuron_index in outputs:
                diff = outputs[neuron_index] > predictions[neuron_index]

                if diff > 0:
                    # neuron firing rate is below the threshold, update neuron morphology to spike sooner
                    # decrease the threshold
                    # NOTE: it will only work for two neurons now consider this action
                    if len(n.threshold) != 2:
                        raise AssertionError
                    if not 0 < diff <= 10:
                        raise AssertionError
                    n.threshold[neuron_index - 1] -= diff
                elif diff < 0:
                    # neuron firing rate is above the threshold, update neuron morphology to spike with lower frequency
                    # increase the threshold
                    if len(n.threshold) != 2:
                        raise AssertionError
                    if not -10 <= diff < 0:
                        raise AssertionError
                    n.threshold[neuron_index - 1] -= diff
                else:
                    #  diff == 0, no action is required
                    pass

            self.prediction_history.clear()
