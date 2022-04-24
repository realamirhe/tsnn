import matplotlib.pyplot as plt
import numpy as np

from src.core.visualizer.history_recorder import HistoryRecorder


class HistoryRecorder1D(HistoryRecorder):
    def plot(self, scale=None):
        if not self.enabled:
            return

        plt.title(self.title)
        if scale is not None:
            plt.plot(np.array(self.history) * scale)
        else:
            plt.plot(self.history)
        plt.show()
        self.reset()
