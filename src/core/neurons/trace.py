import numpy as np

from PymoNNto import Behaviour


class TraceHistory(Behaviour):
    def set_variables(self, n):
        max_delay = self.get_init_attr("max_delay", None, n)
        history_size = 1 if max_delay is None else max_delay + 1
        n.trace = np.zeros((n.size, history_size))

    def new_iteration(self, n):
        latest_trace = n.trace[:, 0].copy()
        n.trace = np.roll(n.trace, 1, axis=1)
        n.trace[:, 0] = latest_trace
