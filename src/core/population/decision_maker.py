import operator

import matplotlib.pyplot as plt
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
)
from src.core.environement.dopamine import DopamineEnvironment
from src.core.environement.inferencer import PhaseDetectorEnvironment
from src.helpers.base import reset_random_seed

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
        self.seed_index = 0
        reset_random_seed(1)

    def new_iteration(self, synapse):
        iteration = synapse.iteration - 1

        neural_activity_plotter.add(
            [ng.A for ng in synapse.network.NeuronGroups if "words" not in ng.tags]
        )

        if iteration in self.outputs:
            if PhaseDetectorEnvironment.is_phase("learning"):
                self.seed_index += 1  # TODO: bug
            reset_random_seed(self.seed_index)
            true_class = self.outputs[iteration]

            pop_activity = {
                ng.tags[0]: ng.A
                for ng in synapse.network.NeuronGroups
                if "words" not in ng.tags
            }

            # TODO: make general and refactor
            if (
                max(pop_activity["pos"], pop_activity["neg"])
                / (min(pop_activity["pos"], pop_activity["neg"]) or 1.0)
                <= self.winner_overcome_ratio
            ):
                DopamineEnvironment.set(0.0)
                winner_class = -1
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
            # reset all the populations n.A files back to zero
            # temp = [
            #     {
            #         ngs.tags[0] + "." + key: getattr(ngs, key)
            #         if key == "A"
            #         else np.mean(getattr(ngs, key))
            #         for key in ngs.resets.keys()
            #     }
            #     for ngs in synapse.network.NeuronGroups
            #     if "words" not in ngs.tags
            # ]
            #
            # if PhaseDetectorEnvironment.is_phase("learning"):
            #     print(
            #         "=== diff ===\n",
            #         [merge_diff(prev, nxt) for prev, nxt in zip(self.temp, temp)],
            #     )
            # self.temp = temp

            for ng in synapse.network.NeuronGroups:
                if "words" not in ng.tags:
                    for key, value in ng.resets.items():
                        if key == "v":
                            ng.v *= 0
                            ng.v += value
                        else:
                            setattr(ng, key, value)

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
            # words_stimulus_plotter.plot()
            dopamine_plotter.plot(splitters=keys)
            # neural_activity.plot(should_reset=False, legend=("pos", "neg"))

            pos_voltage_plotter.plot(splitters=keys)
            neg_voltage_plotter.plot(splitters=keys)
            neural_activity_plotter.plot(splitters=keys)
            activity_plotter.plot(splitters=keys)
            # selected_dw_plotter.plot()
            plt.title("accuracy")
            plt.plot(self.acc)
            plt.show()
