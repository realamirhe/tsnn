class FeatureSwitch:
    def __init__(self, network, features):
        self.features = features
        self.network = network
        self.mode = "train"

    def switch_train(self):
        self.mode = "train"
        self.__switch_features()

    def switch_test(self):
        self.mode = "test"
        self.__switch_features()

    def __switch_features(self):
        self.network.activate_mechanisms([f"{f}:{self.mode}" for f in self.features])
        nonactive_mode = "test" if self.mode == "train" else "train"
        self.network.deactivate_mechanisms(
            [f"{f}:{nonactive_mode}" for f in self.features]
        )


class EpisodeTracker:
    _episode = 0

    @classmethod
    def episode(cls):
        return cls._episode

    @classmethod
    def update(cls):
        cls._episode += 1
