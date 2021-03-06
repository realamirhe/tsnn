from abc import ABC, abstractmethod


class HistoryRecorder(ABC):
    def __init__(
        self,
        title,
        window_size=1,
        vertical_history_separator=False,
        enabled=True,
        should_copy_on_add=False,
        ylim=None,
        save_as_csv=None,
        every_n_episode=None,
    ):
        self.ylim = ylim
        self.history = []
        self.title = title
        self.window_size = window_size
        self.counter = 0
        self.enabled = enabled
        self.history_steps = [] if vertical_history_separator else None
        self.should_copy = should_copy_on_add
        self.save_as_csv = save_as_csv
        self.every_n_episode = every_n_episode

    def add(self, value):
        if not self.enabled:
            return
        if self.counter % self.window_size == 0:
            self.history.append(value if not self.should_copy else value.copy())
        self.counter += 1

    def get(self):
        return self.history

    def configure_plot(self, ylim=None):
        self.ylim = ylim

    def reset(self):
        self.history = []

    @abstractmethod
    def plot(self):
        pass
