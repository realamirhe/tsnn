class DopamineEnvironment:
    dopamine = 0.0

    @classmethod
    def get(cls):
        return cls.dopamine

    @classmethod
    def set(cls, new_dopamine):
        assert -1 <= new_dopamine <= 1
        cls.dopamine = new_dopamine

    @classmethod
    def decay(cls, decay_factor):
        cls.dopamine *= decay_factor
