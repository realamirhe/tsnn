import logging
import operator

import numpy as np
from sklearn.metrics import accuracy_score

from PymoNNto import Behaviour
from src.configs.plotters import (
    dopamine_plotter,
    neural_activity_plotter,
    acc_plotter,
    convergence_plotter,
)
from src.core.environement.dopamine import DopamineEnvironment
from src.core.environement.homeostasis import HomeostasisEnvironment
from src.core.environement.inferencer import PhaseDetectorEnvironment
from src.core.visualizer.words_plotter import wordcloud_plotter

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
        self.wordclouds = {1: [], 0: [], -1: []}

    def new_iteration(self, synapse):
        iteration = synapse.iteration - 1

        neural_activity_plotter.add(
            [ng.A for ng in synapse.network.NeuronGroups if "words" not in ng.tags]
        )
        HomeostasisEnvironment.enabled = False
        if iteration in self.outputs:
            convergence_plotter.add([syn.C for syn in synapse.network.SynapseGroups])
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
                dopamine = 0.0
                # winner_class = random.choice(list(labels2class.values()))
                # dopamine = (-1.0, 1.0)[winner_class == true_class]
                DopamineEnvironment.set(dopamine)
            else:
                # calculate the true of the prediction
                winner_pop = max(pop_activity, key=pop_activity.get)
                winner_class = labels2class[winner_pop]
                # Release or supress dopamine
                dopamine = (-1.0, 1.0)[winner_class == true_class]
                DopamineEnvironment.set(dopamine)

            logging.debug(
                f"Phase={PhaseDetectorEnvironment.phase} => {winner_class=} {true_class=}"
            )

            # Test 19231: Only collect information in leaning phase
            # if PhaseDetectorEnvironment.is_phase("learning"):
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

                self.wordclouds[winner_class].extend(
                    synapse.network.NeuronGroups[0].seen_char
                )
            # toggle the inference phase of the learning
            PhaseDetectorEnvironment.toggle()

        dopamine_plotter.add(DopamineEnvironment.get())
        if synapse.iteration == self.episode_iterations:
            if self._predictions:
                outputs = list(map(operator.itemgetter(1), self._predictions))
                predictions = list(map(operator.itemgetter(0), self._predictions))
                accuracy = accuracy_score(outputs, predictions)
                acc_plotter.add(accuracy)
                logging.info(f"Network accuracy={accuracy} (winner_class, true_class)")
                logging.debug(self._predictions)
                self.acc.append(accuracy)
                self._predictions.clear()

            keys = self.outputs.keys()
            # pos_threshold_plotter.plot(splitters=keys)
            # neg_threshold_plotter.plot(splitters=keys)
            # words_stimulus_plotter.plot(splitters=keys)
            # dopamine_plotter.plot(splitters=keys)
            # pos_base_activity.plot(splitters=keys)
            # neg_base_activity.plot(splitters=keys)
            # pos_voltage_plotter.plot(splitters=keys)
            # neg_voltage_plotter.plot(splitters=keys)
            # neural_activity_plotter.plot(splitters=keys)
            convergence_plotter.plot(
                should_reset=False,
                legend=[syn.tags[0] for syn in synapse.network.SynapseGroups],
            )
            acc_plotter.plot(should_reset=False)

            for title, label in [("neg", 0), ("pos", 1), ("unknown", -1)]:
                sentence = " ".join(list(set(self.wordclouds[label])))
                if not sentence:
                    continue
                wordcloud_plotter(title, sentence)
