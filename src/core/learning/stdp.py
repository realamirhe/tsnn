import numpy as np

from PymoNNto import Behaviour
from src.configs import feature_flags, corpus_config
from src.configs.plotters import (
    dw_plotter,
    w_plotter,
    selected_dw_plotter,
    selected_weights_plotter,
)
from src.core.environement.dopamine import DopamineEnvironment
from src.helpers.base import selected_neurons_from_words


def soft_bound(a_min, A, a_max):
    return (A - a_min) * (a_max - A)


def hard_bound(a_min, A, a_max):
    return np.heaviside(A - a_min, 0) * np.heaviside(a_max - A, 0)


def none_bound(a_min, A, a_max):
    return 1


bounds = {"soft-bound": soft_bound, "hard-bound": hard_bound, "none": none_bound}


class SynapsePairWiseSTDP(Behaviour):
    __slots__ = [
        "a_minus",
        "a_plus",
        "dt",
        "min_delay_threshold",
        "stdp_factor",
        "tau_minus",
        "tau_plus",
        "w_max",
        "w_min",
        "weight_decay",
        "weight_update_strategy",
    ]

    def set_variables(self, synapse):
        synapse.W = synapse.get_synapse_mat("uniform")

        configure = {
            "a_minus": -0.1,
            "a_plus": 0.2,
            "delay_a_minus": -0.1,
            "delay_a_plus": 0.2,
            "delay_factor": 1.0,
            "dt": 1.0,
            "min_delay_threshold": 0.15,
            "stdp_factor": 1.0,
            "tau_minus": 3.0,
            "tau_plus": 3.0,
            "w_max": 10.0,
            "w_min": 0.0,
            "weight_decay": 1.0,
            "max_delay": 1.0,
            "weight_update_strategy": None,
            "delay_update_strategy": None,
        }

        for attr, value in configure.items():
            setattr(self, attr, self.get_init_attr(attr, value, synapse))

        # Scale W from [0,1) to [w_min, w_max)
        synapse.W = synapse.W * (self.w_max - self.w_min) + self.w_min
        synapse.W = np.clip(synapse.W, self.w_min, self.w_max)

        if feature_flags.enable_magic_weights:
            for i, word in enumerate(corpus_config.words):
                indices = [corpus_config.letters.index(char) for char in word]
                synapse.W[:, indices] = self.w_min
                synapse.W[i, indices] = self.w_max

        if self.a_minus >= 0:
            raise AssertionError("a_minus should be negative")

        self.delay_domains = np.arange(synapse.src.size, dtype=int) * np.ones(
            (synapse.dst.size, 1), dtype=int
        )

        if self.weight_update_strategy not in (None, "soft-bound", "hard-bound"):
            raise AssertionError(
                "weight_update_strategy must be one of soft-bound|hard-bound|None"
            )

        if self.delay_update_strategy not in (None, "soft-bound", "hard-bound"):
            raise AssertionError(
                "delay_update_strategy must be one of soft-bound|hard-bound|None"
            )

        selected_weights_plotter.configure_plot(ylim=[self.w_min, self.w_max + 0.2])

    # TODO: add dw_neutral effect into dw_plus
    def new_iteration(self, synapse):
        # For testing only, we won't update synapse weights in test mode!
        if not synapse.recording:
            return

        # add new trace to existing src trace history
        # we don't have access to the latest src asar till here
        # should we accumulate it to the previous last layer trace or just replace that with the latest one
        synapse.src.trace[:, 0] += (
            -synapse.src.trace[:, 0] / self.tau_plus + synapse.src.fired  # dx
        ) * self.dt

        synapse.dst.trace[:, 0] += (
            -synapse.dst.trace[:, 0] / self.tau_minus + synapse.dst.fired  # dy
        ) * self.dt

        #  TODO: coincidence logic detection re-check
        # ltd -> dw_minus (depression) w- d+
        # ltp -> dw_plus (potentiation) w+ d-
        # coincidence -> w ltd & delay ltp
        # dd:= delay_a_plus * ltp + delay_a_minus * ltd
        # dw:= a_plus* ltp + a_min * ltd
        # W += dw
        # delay+=dd
        coincidence = (
            synapse.src.fire_effect.astype(bool) * synapse.dst.fired[:, np.newaxis]
        )
        coincidence = np.logical_not(coincidence)
        ltd = synapse.src.fire_effect * synapse.dst.trace[:, -1][:, np.newaxis]

        delay_ranges = synapse.delay.astype(int)
        mantis = synapse.delay % 1.0
        complement = 1 - mantis

        ltp = (
            (
                synapse.src.trace[self.delay_domains, delay_ranges] * complement
                + synapse.src.trace[
                    self.delay_domains, np.clip(delay_ranges + 1, 0, self.max_delay)
                ]
                * mantis
            )
            # * synapse.src.trace[self.delay_domains, self.delay_ranges+1]*cpomplete ?
            * synapse.dst.fired[:, np.newaxis]
        )

        # soft bound for both delay and stdp separate
        dw = (
            DopamineEnvironment.get()  # from global environment
            * (
                # stdp mechanism
                self.a_plus * ltp
                + self.a_minus * ltd * coincidence
            )
            * bounds[self.weight_update_strategy or "none"](
                self.w_min, synapse.W, self.w_max
            )
            * self.stdp_factor  # stdp scale factor
            * synapse.enabled  # activation of synapse itself
            * self.dt
        )

        dw_plotter.add_image(dw * 1e5)
        rows, cols = selected_neurons_from_words()
        selected_dw_plotter.add(dw[rows, cols])
        synapse.W = synapse.W * self.weight_decay + dw
        synapse.W = np.clip(synapse.W, self.w_min, self.w_max)
        selected_weights_plotter.add(synapse.W[rows, cols])
        w_plotter.add_image(synapse.W, vmin=self.w_min, vmax=self.w_max)

        """ stop condition for delay learning """
        if not feature_flags.enable_delay_update_in_stdp:
            return

        dd = DopamineEnvironment.get() * (
            self.delay_a_plus * ltp * coincidence + self.delay_a_minus * ltd
        )

        # if synapse.src.fire_effect.any() or synapse.dst.fired.any():
        #     print("happy")

        use_shared_delay = dd.shape != synapse.delay.shape
        if use_shared_delay:
            dd = np.mean(dd, axis=0, keepdims=True)

        non_zero_dd = (dd != 0).astype(bool)
        if not non_zero_dd.any():
            return

        should_update = np.min(synapse.delay, axis=1)
        should_update = should_update > self.min_delay_threshold
        synapse.delay[np.logical_not(should_update)] += 1e-5

        if should_update.any():
            synapse.delay += dd * should_update[:, np.newaxis] * self.delay_factor
