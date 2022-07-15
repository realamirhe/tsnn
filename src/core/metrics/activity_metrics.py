from sklearn.metrics import accuracy_score

from PymoNNto import Behaviour


class ActivityMetrics(Behaviour):
    def set_variables(self, n):
        configure = {
            "recording_phase": None,
            "outputs": [],
            "episode_iterations": None,
            "class_index": None,
        }
        for attr, value in configure.items():
            setattr(self, attr, self.get_init_attr(attr, value, n))

        self.size = n.size
        self._predictions = []

    def reset(self):
        self._predictions = []

    # recording is different from input
    def new_iteration(self, n):
        if self.recording_phase is not None and self.recording_phase != n.recording:
            return

        self._predictions.append(int(n.A >= 0.5))
        if n.iteration == self.episode_iterations:
            outputs = []
            predictions = []
            for key, item in self.outputs.items():
                if item == self.class_index:
                    outputs.append(item)
                    predictions.append(self._predictions[key])

            accuracy = accuracy_score(outputs, predictions)
            print(f"Metrics.{n.tags[0]} acc=({accuracy})")
