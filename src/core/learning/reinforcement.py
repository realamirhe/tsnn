import numpy as np

from PymoNNto import Behaviour
from src.core.environement.dopamine import DopamineEnvironment


class Supervisor(Behaviour):
    __slots__ = ["dopamine_decay", "outputs"]

    def set_variables(self, n):
        self.dopamine_decay = 1 - self.get_init_attr("dopamine_decay", 0.0, n)
        self.outputs = self.get_init_attr("outputs", [], n)
        self.current_pattern = 0

    def new_iteration(self, n):
        """
        For post synaptic activity (output) we record the latest desired output in the `self.current_pattern`
        Then check the pre-synaptic activity (prediction)
            - In the first activity (perfect match) we want exact match (L36, L40)
            - In the following activity if it is desirable release reward o.w. punish
            - In the non-active time-step decay the existing dopamine
        """
        if type(self.outputs) is list:
            y_true = self.outputs[n.iteration - 1]
        else:
            y_true = self.outputs.get(n.iteration - 1, None)
        y_pred = n.fired

        if y_true is not None:
            self.current_pattern = np.zeros_like(y_pred)
            self.current_pattern[y_true] = 1

        if np.sum(y_pred) > 0:
            distance = [-1.0, 1.0][int((self.current_pattern == y_pred).all())]
            DopamineEnvironment.set(distance)
        else:
            DopamineEnvironment.decay(self.dopamine_decay)


class ActivitySupervisor(Behaviour):
    __slots__ = ["dopamine_decay", "outputs"]

    def set_variables(self, n):
        self.dopamine_decay = 1 - self.get_init_attr("dopamine_decay", 0.0, n)
        self.class_index = self.get_init_attr("class_index", None, n)
        self.outputs = self.get_init_attr("outputs", [], n)

    def new_iteration(self, n):
        y_pred = int(n.A >= 0.5)

        if n.A != 0:
            distance = [-1.0, 1.0][int(y_pred == self.class_index)]
            DopamineEnvironment.set(distance)
        else:
            DopamineEnvironment.decay(self.dopamine_decay)
