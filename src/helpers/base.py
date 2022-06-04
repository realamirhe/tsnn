import random

import numpy as np


def reset_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def behaviour_generator(behaviours):
    return {index + 1: behaviour for index, behaviour in enumerate(behaviours)}
