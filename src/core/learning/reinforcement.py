import numpy as np
from scipy.spatial.distance import jaccard

from PymoNNto import Behaviour
from src.core.environement.dopamine import DopamineEnvironment


class Supervisor(Behaviour):
    __slots__ = ["dopamine_decay", "outputs"]

    def set_variables(self, neurons):
        self.dopamine_decay = 1 - self.get_init_attr("dopamine_decay", 0.0, neurons)
        self.outputs = self.get_init_attr("outputs", [], neurons)

    def new_iteration(self, neurons):
        output = self.outputs[neurons.iteration - 1]
        prediction = neurons.fired

        if np.isnan(output).any():
            DopamineEnvironment.decay(self.dopamine_decay)
            return

        """ Cosine similarity """
        # distance = 1 - spatial.distance.cosine(
        #     re_range_binary(output), re_range_binary(prediction)
        # )
        # DopamineEnvironment.set(distance or -1)  # replace 0.o effect with -1

        """ mismatch similarity """
        # distance = [-1.0, 1.0][int((output == prediction).all())]
        # DopamineEnvironment.set(distance)

        # DopamineEnvironment.set(-1)
        """ https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jaccard.html """
        distance = jaccard(output, prediction)
        DopamineEnvironment.set(-distance or 1.0)
