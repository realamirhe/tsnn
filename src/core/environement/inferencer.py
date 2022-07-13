class InferenceProvider:
    _is_enabled = False

    def should_freeze_learning(self):
        return not self._is_enabled

    def toggle(self):
        self._is_enabled = not self._is_enabled


InferenceEnvironment = InferenceProvider()
