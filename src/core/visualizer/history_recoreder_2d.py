import os
from glob import glob
from time import localtime

import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.core.visualizer.history_recorder import HistoryRecorder


class HistoryRecorder2D(HistoryRecorder):
    def add_image(self, value, **kwargs):
        if not self.enabled:
            return

        self.add(value)
        # side effect in window size
        if self.counter % self.window_size == 0:
            plt.title(f"{self.title}-{self.counter}")
            # NOTE: clear the memory after figuring
            plt.imshow(self.history.pop(), **kwargs)
            plt.savefig(f"./out/{self.title}-{self.counter}.png")
            self.history.clear()
            plt.clf()

    @staticmethod
    def sort_key(filename):
        return int(filename.split("-")[-1].split(".")[0])

    def plot(self):
        if not self.enabled:
            return

        time = localtime()
        with imageio.get_writer(
            f"./out/{self.title} {time.tm_hour}-{time.tm_min}-{time.tm_sec}.gif",
            mode="I",
        ) as writer:
            frames = glob(f"./out/{self.title}-*.png")
            frames.sort(key=HistoryRecorder2D.sort_key)
            for filename in tqdm(frames, f"generating {self.title} gif"):
                image = imageio.imread(filename)
                writer.append_data(image)

        for filename in tqdm(
            glob(f"./out/{self.title}-*.png"),
            f"removing {self.title} generated frames",
        ):
            os.remove(filename)
        self.reset()
