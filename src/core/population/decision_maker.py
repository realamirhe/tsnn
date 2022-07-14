import operator

from sklearn.metrics import accuracy_score

from PymoNNto import Behaviour
from src.configs.plotters import (
    neg_threshold_plotter,
    pos_threshold_plotter,
    selected_dw_plotter,
)
from src.core.environement.dopamine import DopamineEnvironment
from src.core.environement.inferencer import InferenceEnvironment

labels2class = {"pos": 1, "neg": 0}


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

    def new_iteration(self, synapse):
        if synapse.iteration in self.outputs:
            true_class = self.outputs[synapse.iteration]

            pop_activity = {
                ng.tags[0]: ng.A
                for ng in synapse.network.NeuronGroups
                if "words" not in ng.tags
            }

            # TODO: make general and refactor
            if (
                abs(pop_activity["pos"] - pop_activity["neg"])
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

            # reset all the populations n.A files back to zero
            for ng in synapse.network.NeuronGroups:
                if "words" not in ng.tags:
                    for key, value in ng.resets.items():
                        setattr(ng, key, value)

            if not InferenceEnvironment.should_freeze_learning():
                self._predictions.append((winner_class, true_class))
            # toggle the inference phase of the learning
            InferenceEnvironment.toggle()

        if synapse.iteration == self.episode_iterations:
            outputs = list(map(operator.itemgetter(1), self._predictions))
            predictions = list(map(operator.itemgetter(0), self._predictions))
            accuracy = accuracy_score(outputs, predictions)
            print(f"Network accuracy={accuracy}")
            self._predictions.clear()

            pos_threshold_plotter.plot()
            neg_threshold_plotter.plot()
            # selected_dw_plotter.plot()
