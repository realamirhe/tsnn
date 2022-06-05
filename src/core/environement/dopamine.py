class DopamineEnvironment:
    _dopamine = 0.0

    @classmethod
    def get(cls):
        return cls._dopamine

    @classmethod
    def set(cls, new_dopamine):
        if not -1 <= new_dopamine <= 1:
            raise AssertionError
        cls._dopamine = new_dopamine

    @classmethod
    def decay(cls, decay_factor):
        cls._dopamine *= decay_factor
