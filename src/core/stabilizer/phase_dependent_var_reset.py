from PymoNNto import Behaviour

from src.core.environement.inferencer import PhaseDetectorEnvironment
from src.helpers.base import reset_random_seed


class PhaseDependentVarReset(Behaviour):
    def set_variables(self, n):
        self.phase = None

        self.should_reset_seed = self.get_init_attr("should_reset_seed", False, n)
        if self.should_reset_seed:
            self.repeat_seed = 0
            reset_random_seed(self.repeat_seed)

    def new_iteration(self, n):
        # Phase changing timestep
        if self.phase != PhaseDetectorEnvironment.phase:
            self.phase = PhaseDetectorEnvironment.phase

            if self.should_reset_seed:
                # Only increase the seed on second and later inference phase!
                self.repeat_seed += int(PhaseDetectorEnvironment.is_phase("inference"))
                reset_random_seed(self.repeat_seed)

            # Reset Accumulated population traces
            n.trace *= 0

            # Reset neuron local variables
            n.A = 0.0
            n.fired[:] = False
            n.v[:] = n.v_rest
