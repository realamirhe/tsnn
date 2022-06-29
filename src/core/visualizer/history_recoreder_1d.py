import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.core.visualizer.history_recorder import HistoryRecorder
from src.helpers.network import EpisodeTracker


class HistoryRecorder1D(HistoryRecorder):
    def plot(self, scale=None, should_reset=True, legend=None):
        if not self.enabled:
            return

        plt.title(self.title + f" (eps={EpisodeTracker.episode()})")

        if self.save_as_csv is not None:
            pd.DataFrame(
                np.array(self.history) * scale
                if scale is not None
                else np.array(self.history)
            ).to_csv(f"out/csv/{self.title}-{EpisodeTracker.episode()}.csv")

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

        if self.ylim is not None:
            plt.gca().set_ylim(self.ylim)

        if (
            self.every_n_episode is None
            or EpisodeTracker.episode() % self.every_n_episode == 0
        ):
            plt.show()
        else:
            plt.clf()

        if should_reset:
            self.reset()
