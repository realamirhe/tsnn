import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from PymoNNto import Behaviour
from src.core.nlp.constants import UNK


class Metrics(Behaviour):
    # fmt: off
    __slots__ = ["recording_phase", "outputs", "_old_recording", "_predictions", "words"]

    # fmt: on
    def set_variables(self, n):
        configure = {"recording_phase": None, "outputs": [], "words": []}
        for attr, value in configure.items():
            setattr(self, attr, self.get_init_attr(attr, value, n))

        self._old_recording = n.recording
        self._predictions = []

    def reset(self):
        self._predictions = []

    # recording is different from input
    def new_iteration(self, n):
        if self.recording_phase is not None and self.recording_phase != n.recording:
            return

        if not np.isnan(self.outputs[n.iteration - 1]).any():
            # TODO: can append the int here also
            self._predictions.append(n.fired)  # [T, F]

        if n.iteration == len(self.outputs):
            bit_range = 1 << np.arange(self.outputs[0].size)

            presentation_words = self.words + [UNK]
            outputs = [o.dot(bit_range) for o in self.outputs if not np.isnan(o).any()]
            predictions = [p.dot(bit_range) for p in self._predictions]

            network_phase = "Testing" if "test" in self.tags[0] else "Training"
            accuracy = accuracy_score(outputs, predictions)

            precision = precision_score(outputs, predictions, average="micro")
            f1 = f1_score(outputs, predictions, average="micro")
            recall = recall_score(outputs, predictions, average="micro")
            # confusion matrix
            cm = confusion_matrix(outputs, predictions)
            cm_sum = cm.sum(axis=1)

            frequencies = np.asarray(np.unique(outputs, return_counts=True)).T
            frequencies_p = np.asarray(np.unique(predictions, return_counts=True)).T

            print(
                "---" * 15,
                f"{network_phase}",
                f"accuracy: {accuracy}",
                f"precision: {precision}",
                f"f1: {f1}",
                f"recall: {recall}",
                f"{','.join(presentation_words)} = {cm.diagonal() / np.where(cm_sum > 0, cm_sum, 1)}",
                "---" * 15,
                f"[Output] frequencies::\n{frequencies}",
                f"[Prediction] frequencies::\n{frequencies_p}",
                sep="\n",
                end="\n\n",
            )

            # display_labels=['none', 'abc', 'omn', 'both']
            cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
            cm_display.plot()
            plt.title(f"{network_phase} Confusion Matrix")
            plt.show()
