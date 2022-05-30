from abc import ABC, abstractmethod


class HistoryRecorder(ABC):
    def __init__(
        self, title, window_size=1, vertical_history_separator=False, enabled=True
    ):
        self.history = []
        self.title = title
        self.window_size = window_size
        self.counter = 0
        self.enabled = enabled
        self.history_steps = [] if vertical_history_separator else None

    def add(self, value, should_copy=False):
        if not self.enabled:
            return
        if self.counter % self.window_size == 0:
            self.history.append(value if not should_copy else value.copy())
        self.counter += 1

    def get(self):
        return self.history

    def reset(self):
        self.history = []

    @abstractmethod
    def plot(self):
        pass
