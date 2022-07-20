import numpy as np


class Homeostasis:
    enabled = False

    def __init__(self, sentence_nums, class_nums):
        self.history = np.zeros((class_nums, sentence_nums))
        self.inserted_pops_count = 0

    @property
    def num_sentences(self):
        return self.history.shape[1]

    @property
    def has_enough_data(self):
        return self.inserted_pops_count >= self.num_sentences

    @property
    def accumulated_activities(self):
        return np.sum(self.history, axis=1)

    def add_pop_activity(self, pop_activity):
        self.inserted_pops_count += 1
        self.history[:, 1:] = self.history[:, 0:-1]
        self.history[:, 0] = pop_activity


HomeostasisEnvironment = Homeostasis(sentence_nums=4, class_nums=2)
