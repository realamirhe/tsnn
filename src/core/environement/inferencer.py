from typing import Union, Literal


class PhaseDetector:
    phase = "inference"
    learning = "learning"
    inference = "inference"

    def is_phase(
        self, validation_phase: Union[Literal["learning"], Literal["inference"]]
    ):
        if validation_phase not in (self.inference, self.learning):
            raise AssertionError(
                f"is_phase only accept {self.inference} or {self.learning}"
            )
        return self.phase == validation_phase

    def toggle(self):
        if self.phase == self.inference:
            self.phase = self.learning
        else:
            self.phase = self.inference


PhaseDetectorEnvironment = PhaseDetector()
