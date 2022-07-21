import operator

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

from PymoNNto import Behaviour
from src.configs.plotters import (
    dopamine_plotter,
    pos_voltage_plotter,
    neg_voltage_plotter,
    neural_activity_plotter,
    pos_threshold_plotter,
    neg_threshold_plotter,
    activity_plotter,
    words_stimulus_plotter,
)
from src.core.environement.dopamine import DopamineEnvironment
from src.core.environement.homeostasis import HomeostasisEnvironment
from src.core.environement.inferencer import PhaseDetectorEnvironment

labels2class = {"pos": 1, "neg": 0}


def merge_diff(prev, nxt):
    return {key: nxt[key] - prev[key] for key in prev.keys()}


class NetworkDecisionMaker(Behaviour):
    def set_variables(self, synapse):
        configure = {
            "outputs": None,
            "episode_iterations": None,
            "winner_overcome_ratio": 0,
        }

        for attr, value in configure.items():
            setattr(self, attr, self.get_init_attr(attr, value, synapse))

        self._predictions = []
        self.acc = []

    def new_iteration(self, synapse):
        iteration = synapse.iteration - 1

        neural_activity_plotter.add(
            [ng.A for ng in synapse.network.NeuronGroups if "words" not in ng.tags]
        )

        HomeostasisEnvironment.enabled = False
        if iteration in self.outputs:
            true_class = self.outputs[iteration]

            class_populations = [
                ng for ng in synapse.network.NeuronGroups if "words" not in ng.tags
            ]
            #  NOTE: python 3.7 granted to have ordered dict, GOD please help us
            pop_activity = {ng.tags[0]: ng.A for ng in class_populations}

            # TODO: make general and refactor
            if (
                max(pop_activity["pos"], pop_activity["neg"])
                / (min(pop_activity["pos"], pop_activity["neg"]) or 1.0)
                <= self.winner_overcome_ratio
            ):
                winner_class = -1
                DopamineEnvironment.set(0.0)
            else:
                # calculate the true of the prediction
                winner_pop = max(pop_activity, key=pop_activity.get)
                winner_class = labels2class[winner_pop]
                # Release or supress dopamine
                dopamine = (-1.0, 1.0)[winner_class == true_class]
                DopamineEnvironment.set(dopamine)

            print(
                f"Phase={PhaseDetectorEnvironment.phase} => {winner_class=} {true_class=}"
            )

            HomeostasisEnvironment.add_pop_activity(
                np.array(list(pop_activity.values()))
            )

            # S
            base_activities = HomeostasisEnvironment.accumulated_activities
            base_activities /= np.array([ng.size for ng in class_populations])
            base_activities /= HomeostasisEnvironment.num_sentences
            # base_activity = np.average(base_activity)
            HomeostasisEnvironment.enabled = True

            for base_activity, ng in zip(base_activities, class_populations):
                ng.base_activity = base_activity

            if PhaseDetectorEnvironment.is_phase("learning"):
                self._predictions.append((winner_class, true_class))
            # toggle the inference phase of the learning
            PhaseDetectorEnvironment.toggle()

        dopamine_plotter.add(DopamineEnvironment.get())
        if synapse.iteration == self.episode_iterations:
            if self._predictions:
                outputs = list(map(operator.itemgetter(1), self._predictions))
                predictions = list(map(operator.itemgetter(0), self._predictions))
                accuracy = accuracy_score(outputs, predictions)
                print(f"Network accuracy={accuracy} (winner_class, true_class)")
                print(self._predictions)
                self.acc.append(accuracy)
                self._predictions.clear()

            keys = self.outputs.keys()
            pos_threshold_plotter.plot(splitters=keys)
            neg_threshold_plotter.plot(splitters=keys)
            words_stimulus_plotter.plot(splitters=keys)
            dopamine_plotter.plot(splitters=keys)
            # neural_activity.plot(should_reset=False, legend=("pos", "neg"))

            pos_voltage_plotter.plot(splitters=keys)
            neg_voltage_plotter.plot(splitters=keys)
            neural_activity_plotter.plot(splitters=keys)
            # selected_dw_plotter.plot()
            plt.title("accuracy")
            plt.plot(self.acc)
            plt.show()
