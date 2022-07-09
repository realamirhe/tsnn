import numpy as np


def entity2spike(entity, entities):
    spike = np.zeros(len(entities), dtype=int)
    if entity in entities:
        spike[entities.index(entity)] = 1
    return spike
