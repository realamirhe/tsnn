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
from src.configs import feature_flags, corpus_config
from src.configs.corpus_config import UNK
from src.configs.plotters import (
    dw_plotter,
    w_plotter,
    dopamine_plotter,
    threshold_plotter,
    delay_plotter,
    activity_plotter,
    words_stimulus_plotter,
    selected_delay_plotter,
    selected_dw_plotter,
    selected_weights_plotter,
    dst_firing_plotter,
)
from src.core.environement.dopamine import DopamineEnvironment
from src.helpers.network import EpisodeTracker


class Metrics(Behaviour):
    # fmt: off
    __slots__ = ["recording_phase", "outputs", "_predictions", "words", "episode_iterations"]

    # fmt: on
    def set_variables(self, n):
        configure = {
            "recording_phase": None,
            "outputs": [],
            "words": [],
            "episode_iterations": None,
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

        # if not np.isnan(self.outputs[n.iteration - 1]).any():
        #     # NOTE: 🚀 can append the int here also
        self._predictions.append(n.fired.copy())
        dopamine_plotter.add(DopamineEnvironment.get())

        if n.iteration == self.episode_iterations:
            dw_plotter.plot()
            w_plotter.plot()
            legend = list("".join(corpus_config.words))
            selected_delay_plotter.plot(legend=legend, should_reset=False)
            selected_weights_plotter.plot(legend=legend, should_reset=False)
            selected_dw_plotter.plot(legend=legend, should_reset=False)
            dopamine_plotter.plot(should_reset=False)
            threshold_plotter.plot(legend=corpus_config.words, should_reset=False)
            delay_plotter.plot()
            activity_plotter.plot(should_reset=False)
            words_stimulus_plotter.plot()
            dst_firing_plotter.plot(should_reset=False)

            presentation_words = self.words + [UNK]

            # Full confusion matrix plot
            outputs = list(self.outputs.values())
            predictions = [
                np.argmax(self._predictions[timestep])
                if np.sum(self._predictions[timestep])
                else -1
                for timestep in self.outputs.keys()
            ]

            network_phase = "Testing" if "test" in self.tags[0] else "Training"
            accuracy = accuracy_score(outputs, predictions)

            precision = precision_score(outputs, predictions, average="micro")
            f1 = f1_score(outputs, predictions, average="micro")
            recall = recall_score(outputs, predictions, average="micro")

            cm = confusion_matrix(outputs, predictions)
            cm_sum = cm.sum(axis=1)
            frequencies = np.asarray(np.unique(outputs, return_counts=True)).T
            frequencies_p = np.asarray(np.unique(predictions, return_counts=True)).T

            if feature_flags.enable_metric_logs:
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
                predictions = np.array(predictions)
                outputs = np.array(outputs)
                print(
                    "prediction",
                    np.sum(predictions == 0),
                    np.sum(predictions == 1),
                    np.sum(predictions == 2),
                )
                print(
                    "output",
                    np.sum(outputs == 0),
                    np.sum(outputs == 1),
                    np.sum(outputs == 2),
                )
                print("==========")

            if feature_flags.enable_cm_plot:
                cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
                cm_display.plot()
                plt.title(
                    f"{network_phase} Confusion Matrix "
                    f"(episode={EpisodeTracker.episode()})"
                )
                plt.show()
