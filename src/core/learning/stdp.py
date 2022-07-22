import logging

import numpy as np

from PymoNNto import Behaviour
from src.configs import feature_flags, corpus_config
from src.configs.corpus_config import letters
from src.configs.plotters import (
    selected_weights_plotter,
)
from src.core.environement.dopamine import DopamineEnvironment
from src.core.environement.inferencer import PhaseDetectorEnvironment


def soft_bound(a_min, A, a_max):
    return (A - a_min) * (a_max - A)


def hard_bound(a_min, A, a_max):
    return np.heaviside(A - a_min, 0) * np.heaviside(a_max - A, 0)


def none_bound(a_min, A, a_max):
    return 1


bounds = {"soft-bound": soft_bound, "hard-bound": hard_bound, "none": none_bound}

LET = np.array(list(letters))


class SynapsePairWiseSTDP(Behaviour):
    def set_variables(self, synapse):

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
            "max_delay": 1.0,
            "weight_update_strategy": None,
            "delay_update_strategy": None,
        }

        for attr, value in configure.items():
            setattr(self, attr, self.get_init_attr(attr, value, synapse))

        # Scale W from [0,1) to [w_min, w_max)
        synapse.W = synapse.get_synapse_mat("uniform")
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
        logging.info(
            f"""
            tags={synapse.tags[0]}
            W ≈ {np.round(np.average(synapse.W), 1)}
            -------------
            W´ = {np.round(np.sum(synapse.W != 0) / synapse.W.size * 100, 1)}
            W ∈ [{self.w_min}, {self.w_max}] 
            """
        )

    # TODO: add dw_neutral effect into dw_plus
    def new_iteration(self, synapse):

        synapse.C = np.average(soft_bound(self.w_min, synapse.W, self.w_max))

        # For testing only, we won't update synapse weights in test mode!
        if not synapse.recording or PhaseDetectorEnvironment.is_phase("inference"):
            return

        #  TODO: coincidence logic detection re-check
        # add new trace to existing src trace history
        # we don't have access to the latest src asar till here
        # should we accumulate it to the previous last layer trace or just replace that with the latest one
        # ltd -> dw_minus (depression) w- d+
        # ltp -> dw_plus (potentiation) w+ d-
        # coincidence -> w ltd & delay ltp
        # dd:= delay_a_plus * ltp + delay_a_minus * ltd
        # dw:= a_plus* ltp + a_min * ltd
        # W += dw
        # delay+=dd
        coincidence = (
            synapse.src_fire_effect.astype(bool) * synapse.dst.fired[:, np.newaxis]
        )
        non_coincidence = np.logical_not(coincidence)
        ltd = synapse.src_fire_effect * synapse.dst.trace[:, 0][:, np.newaxis]

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
                + self.a_minus * ltd * non_coincidence
            )
            * bounds[self.weight_update_strategy or "none"](
                self.w_min, synapse.W, self.w_max
            )
            * self.stdp_factor  # stdp scale factor
            * synapse.enabled  # activation of synapse itself
            * self.dt
        )

        synapse.W[synapse.W > 0.01] -= 1e-5
        synapse.delay[synapse.delay < self.max_delay - 0.01] += 1e-5

        synapse.W = synapse.W + dw
        synapse.W = np.clip(synapse.W, self.w_min, self.w_max)

        """ stop condition for delay learning """
        if not feature_flags.enable_delay_update_in_stdp:
            return

        dd = DopamineEnvironment.get() * (
            self.delay_a_plus * ltp * non_coincidence + self.delay_a_minus * ltd
        )

        use_shared_delay = dd.shape != synapse.delay.shape
        if use_shared_delay:
            dd = np.mean(dd, axis=0, keepdims=True)

        non_zero_dd = (dd != 0).astype(bool)
        if not non_zero_dd.any():
            return

        should_update = np.min(synapse.delay, axis=1)
        should_update = should_update > self.min_delay_threshold
        # synapse.delay[np.logical_not(should_update)] += 1e-5

        if should_update.any():
            synapse.delay += dd * should_update[:, np.newaxis] * self.delay_factor
