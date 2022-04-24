from abc import ABC, abstractmethod


class HistoryRecorder(ABC):
    def __init__(self, title, window_size=1, enabled=True):
        self.history = []
        self.title = title
        self.window_size = window_size
        self.counter = 0
        self.enabled = enabled

    def add(self, value):
        if not self.enabled:
            return
        self.counter += 1
        if self.counter % self.window_size == 0:
            self.history.append(value)

    def get(self):
        return self.history

    def reset(self):
        self.history = []

    @abstractmethod
    def plot(self):
        pass
