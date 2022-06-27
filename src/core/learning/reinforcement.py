import numpy as np

from PymoNNto import Behaviour
from src.core.environement.dopamine import DopamineEnvironment


class Supervisor(Behaviour):
    """
    - a -> decay
    - ab -> decay
    - abc -> decay
    - abc_ -> validation_mechanism (reward|punishment)
    """

    __slots__ = ["dopamine_decay", "outputs"]

    def set_variables(self, n):
        self.dopamine_decay = 1 - self.get_init_attr("dopamine_decay", 0.0, n)
        self.outputs = self.get_init_attr("outputs", [], n)
        self.current_pattern = self.outputs[0] * np.nan

    def new_iteration(self, n):
        """
        For post synaptic activity (output) we record the latest desired output in the `self.current_pattern`
        Then check the pre-synaptic activity (prediction)
            - In the first activity (perfect match) we want exact match (L36, L40)
            - In the following activity if it is desirable release reward o.w. punish
            - In the non-active time-step decay the existing dopamine
        """

        output = self.outputs[n.iteration - 1]
        prediction = n.fired

        if not np.isnan(output).any():
            self.current_pattern = output

            if np.sum(prediction) == 0:
                DopamineEnvironment.set(-1)
                return

        if np.sum(prediction) > 0:
            distance = [-1.0, 1.0][int((self.current_pattern == prediction).all())]
            DopamineEnvironment.set(distance)
        else:
            DopamineEnvironment.decay(self.dopamine_decay)

        # Cosine similarity
        # distance = 1 - spatial.distance.cosine(output + 1, prediction + 1)
        # DopamineEnvironment.set(distance or -1)  # replace 0.o effect with -1

        # Mismatch similarity
        # distance = [-1.0, 1.0][int((output == prediction).all())]
        # DopamineEnvironment.set(distance)

        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jaccard.html
        # distance = jaccard(output, prediction)
        # DopamineEnvironment.set(-distance or 1.0)
