import matplotlib.pyplot as plt
import numpy as np

from src.core.visualizer.history_recorder import HistoryRecorder


class HistoryRecorder1D(HistoryRecorder):
    def plot(self, scale=None, should_reset=True, legend=None):
        if not self.enabled:
            return

        plt.title(self.title)
        if scale is not None:
            plt.plot(np.array(self.history) * scale)
        else:
            plt.plot(self.history)

        if legend is not None:
            plt.gca().legend(legend)

        plt.show()

        if should_reset:
            self.reset()
