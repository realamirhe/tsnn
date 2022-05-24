import numpy as np

from PymoNNto import Behaviour


class TraceHistory(Behaviour):
    __slots__ = ["trace_decay_factor"]

    def set_variables(self, n):
        max_delay = self.get_init_attr("max_delay", None, n)
        history_size = 1 if max_delay is None else max_delay + 1
        n.trace = np.zeros((n.size, history_size))
        self.trace_decay_factor = self.get_init_attr("trace_decay_factor", 1, n)

    def new_iteration(self, n):
        latest_trace = n.trace[:, -1] * self.trace_decay_factor
        n.trace = np.roll(n.trace, -1, axis=1)
        n.trace[:, -1] = latest_trace
