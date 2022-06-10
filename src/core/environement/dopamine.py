class DopamineEnvironment:
    _dopamine = 0.0

    def get(self):
        return self._dopamine

    def set(self, new_dopamine):
        if not -1 <= new_dopamine <= 1:
            raise AssertionError
        self._dopamine = new_dopamine

    def decay(self, decay_factor):
        self._dopamine *= decay_factor


dopamine = DopamineEnvironment()
