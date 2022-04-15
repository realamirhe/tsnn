import numpy as np
from scipy.spatial.distance import jaccard

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

    def new_iteration(self, n):
        output = self.outputs[n.iteration - 1]
        prediction = n.fired  # [T, F]

        if np.isnan(output).any():
            DopamineEnvironment.decay(self.dopamine_decay)
            return

        """ Cosine similarity """
        # distance = 1 - spatial.distance.cosine(
        #     re_range_binary(output), re_range_binary(prediction)
        # )
        # DopamineEnvironment.set(distance or -1)  # replace 0.o effect with -1

        """ mismatch similarity """
        distance = [-1.0, 1.0][int((output == prediction).all())]
        DopamineEnvironment.set(distance)
        # [F, F]
        # [T, F]

        # DopamineEnvironment.set(-1)
        """ https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jaccard.html """
        # distance = jaccard(output, prediction)
        # DopamineEnvironment.set(-distance or 1.0)
