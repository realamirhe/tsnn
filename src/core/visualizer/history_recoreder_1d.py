import matplotlib.pyplot as plt
import numpy as np

from src.core.visualizer.history_recorder import HistoryRecorder
from src.helpers.network import EpisodeTracker


class HistoryRecorder1D(HistoryRecorder):
    def plot(self, scale=None, should_reset=True, legend=None):
        if not self.enabled:
            return

        plt.title(self.title + f" (eps={EpisodeTracker.episode()})")
        if scale is not None:
            plt.plot(np.array(self.history) * scale)
        else:
            plt.plot(self.history)

        if legend is not None:
            plt.gca().legend(legend)

        if not should_reset and self.history_steps is not None:
            self.history_steps.append(len(self.history))
            for x in self.history_steps:
                plt.axvline(x, color="b", linestyle="--", alpha=0.3)

        plt.show()

        if should_reset:
            self.reset()
