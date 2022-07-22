import logging

from PymoNNto import Network, NeuronGroup, SynapseGroup
from examples.IMDB.config import words_max_delay, population_size
from examples.IMDB.data_engineering.spike_generator import (
    words2spikes,
    joined_corpus_maker,
    sentence2spikes,
)
from examples.IMDB.data_engineering.test_train_dataset import test_train_dataset
from examples.IMDB.data_engineering.words_extractor import extract_words
from examples.IMDB.network.nueron_morphology import (
    get_base_neuron_config,
    get_base_homeostasis,
)
from examples.IMDB.network.synapse_morphology import (
    get_base_delay,
    get_weight_stdp,
    get_base_stdp,
    get_base_delay_stdp,
)
from src.core.learning.delay import SynapseDelay as FireHistorySynapseDelay
from src.core.learning.stdp import SynapsePairWiseSTDP
from src.core.learning.stdp_delayless import SynapsePairWiseSTDPWithoutDelay
from src.core.neurons.current import CurrentStimulus
from src.core.neurons.neurons import StreamableLIFNeurons
from src.core.neurons.trace import TraceHistory
from src.core.population.decision_maker import NetworkDecisionMaker
from src.core.population.pop_activity import PopulationBaseActivity
from src.core.stabilizer.phase_dependent_var_reset import PhaseDependentVarReset
from src.core.stabilizer.pop_activity_base_homeostasis import (
    PopulationActivityBaseHomeostasis,
)
from src.helpers.base import reset_random_seed
from src.helpers.corpus import replicate_df_rows
from src.helpers.network import EpisodeTracker

"""
DONE: Check the correct path for synapse_lens_selector
DONE: For population we don't need winner take all, need activity measuring instead
DONE: For population we don't need supervisor for activity
DONE: Remove delay learning for the (pos|neg) recurrent synapse connections
DONE: Add weights initializer with J/N strategies

TODO: 🥋 Critical, Current might need a change for the population lenses synapse parameter
TODO: 🥋 Reset mechanism in delay learning for the sentence layer are specific to their echo system make a copy and use different behvaiours


TODO: Check the inhibitory connection (does it need to be negative, is negative w_min works fine?)
TODO: Review Brunel Hakim source code for J setter on the neuron group
TODO: The `dt` in the TraceHistory is removed because it is 1
TODO: Make general decision-maker for more than one in neural population 🔥

Ques: should n.A be divided by #pop and #pop_items?
Ques: Supervisor should not be for every connection, they effect dopamine!
Ques: Different Metrics must also be excluded or combine their reports!
Ques: Check the ceiling effect on homeostasis behaviour (PopHomeostasis)
Ques: Is it meaningful to reduce max weight so we get 51,49 rule of activation in the population of pos or neg?
Ques: Should we use src.fired or src.fire_effect in the `pop-activity` -> PopulationBaseActivity
Ques: How we should handle the reinforcement for the sentences. they are really long and class based
Ques: Random connection removing in synapse.W
Ques: Double check weight initialization J/√PN
Ques: What does max_word_delay means in the words:pos connection Q01
Ques: How we should fill the gap of decay effect in the words layer Q02
Ques: Should ActivitySupervisor needs to know how other pop behave or not?
Ques: Now all words are selected but we might need to trim them in `word_length_threshold`? Should we add -1 in the words layer



to end or end/2
DONE: In neuron morphology `tau` -> max(words_spacing_gap, maximum_length_words), max_length is different from what it should be

IMPROVE: max instead of clip and slice instead of roll
IMPROVE: Resolve and fix the `self.outputs` in the reinforcement

Pos words, 
Neg words

-> {select:5} => sentences
"""


def network():
    logging.basicConfig(level=logging.WARNING)

    network = Network()
    # 🔥 NOTE: the farc of train set is not 1 anymore, resolve that before run
    (train_df, _) = test_train_dataset(train_size=20, random_state=42)
    common_words = extract_words(train_df, word_length_threshold=10)
    # duplicate each sentence (inference + learning) step
    train_df = replicate_df_rows(train_df)
    words_stream = words2spikes(train_df, common_words)
    sentence_stream = sentence2spikes(train_df)
    joined_corpus = joined_corpus_maker(train_df)
    simulation_iterations = len(words_stream)

    # 🔥 NOTE: Initial threshold has been set low for testing purpose
    lif_base = get_base_neuron_config()
    homeostasis_base = get_base_homeostasis()
    # 🐓: Q01
    delay_args = get_base_delay()
    # 🐓: Q02
    stdp_weights_args = get_weight_stdp()
    stdp_args = get_base_stdp()
    stdp_delay_args = get_base_delay_stdp()
    balanced_network_args = {"J": 100, "P": 0.5}

    n_episodes = 1000

    # Words neuron group
    reset_random_seed(1000)
    words_ng = NeuronGroup(
        net=network,
        tag="words",
        size=len(common_words),
        behaviour={
            1: StreamableLIFNeurons(
                stream=words_stream, joined_corpus=joined_corpus, **lif_base
            ),
            2: TraceHistory(max_delay=words_max_delay, tau=4.0),
        },
    )

    # positive population neuron groups
    pos_pop_ng = NeuronGroup(
        net=network,
        tag="pos",
        size=population_size,
        behaviour={
            1: PhaseDependentVarReset(),
            2: CurrentStimulus(
                adaptive_noise_scale=0.9,
                noise_scale_factor=0.01,
                stimulus_scale_factor=0.03,
                synapse_lens_selector=[
                    {"path": ["words:pos", 0], "type": "delayed"},
                    {"path": ["pos:pos", 0], "type": "normal"},
                    {"path": ["neg:pos", 0], "type": "normal"},
                ],
            ),
            3: StreamableLIFNeurons(**lif_base, has_long_term_effect=True),
            4: TraceHistory(max_delay=words_max_delay, tau=4.0),
            5: PopulationBaseActivity(tag="pop-activity"),
            8: PopulationActivityBaseHomeostasis(**homeostasis_base),
        },
    )

    # negative population neuron groups
    neg_pop_ng = NeuronGroup(
        net=network,
        tag="neg",
        size=population_size,
        behaviour={
            1: PhaseDependentVarReset(should_reset_seed=True),
            2: CurrentStimulus(
                adaptive_noise_scale=0.9,
                noise_scale_factor=0.01,
                stimulus_scale_factor=0.03,
                synapse_lens_selector=[
                    {"path": ["words:neg", 0], "type": "delayed"},
                    {"path": ["neg:neg", 0], "type": "normal"},
                    {"path": ["pos:neg", 0], "type": "normal"},
                ],
            ),
            3: StreamableLIFNeurons(**lif_base, has_long_term_effect=True),
            4: TraceHistory(max_delay=words_max_delay, tau=4.0),
            5: PopulationBaseActivity(tag="pop-activity"),
            8: PopulationActivityBaseHomeostasis(**homeostasis_base),
        },
    )

    # words -> pos_pop_ng
    SynapseGroup(
        net=network,
        src=words_ng,
        dst=pos_pop_ng,
        tag="words:pos",
        behaviour={
            1: FireHistorySynapseDelay(**delay_args),
            6: SynapsePairWiseSTDP(**stdp_args, **stdp_delay_args, **stdp_weights_args),
        },
    )
    # words -> neg_pop_ng
    SynapseGroup(
        net=network,
        src=words_ng,
        dst=neg_pop_ng,
        tag="words:neg",
        behaviour={
            1: FireHistorySynapseDelay(**delay_args),
            6: SynapsePairWiseSTDP(**stdp_args, **stdp_delay_args, **stdp_weights_args),
        },
    )
    # pos_pop_ng -> pos_pop_ng
    SynapseGroup(
        net=network,
        src=pos_pop_ng,
        dst=pos_pop_ng,
        tag="pos:pos",
        behaviour={
            6: SynapsePairWiseSTDPWithoutDelay(
                **stdp_args, **stdp_weights_args, **balanced_network_args
            ),
        },
    )
    # neg_pop_ng -> neg_pop_ng
    SynapseGroup(
        net=network,
        src=neg_pop_ng,
        dst=neg_pop_ng,
        tag="neg:neg",
        behaviour={
            6: SynapsePairWiseSTDPWithoutDelay(
                **stdp_args, **stdp_weights_args, **balanced_network_args
            ),
        },
    )
    # pos_pop_ng -> neg_pop_ng
    SynapseGroup(
        net=network,
        src=pos_pop_ng,
        dst=neg_pop_ng,
        tag="pos:neg",
        behaviour={
            6: SynapsePairWiseSTDPWithoutDelay(
                **stdp_args,
                P=balanced_network_args["P"] - 0.1,
                J=-110,
                is_inhibitory=True,
                w_min=-stdp_weights_args["w_max"],
                w_max=0
            ),
        },
    )
    # neg_pop_ng -> pos_pop_ng
    SynapseGroup(
        net=network,
        src=neg_pop_ng,
        dst=pos_pop_ng,
        tag="neg:pos",
        behaviour={
            6: SynapsePairWiseSTDPWithoutDelay(
                **stdp_args,
                P=balanced_network_args["P"] + 0.1,
                J=-90,
                is_inhibitory=True,
                w_min=-stdp_weights_args["w_max"],
                w_max=0
            ),
            7: NetworkDecisionMaker(
                outputs=sentence_stream,
                episode_iterations=simulation_iterations,
                winner_overcome_ratio=1.5,
            ),
        },
    )

    network.initialize(info=False)

    """ TRAINING """
    for _ in range(n_episodes):
        EpisodeTracker.update()
        network.iteration = 0
        network.simulate_iterations(simulation_iterations, measure_block_time=False)


if __name__ == "__main__":
    network()
