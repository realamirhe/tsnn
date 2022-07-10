from tqdm import tqdm

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
from src.core.learning.reinforcement import ActivitySupervisor
from src.core.learning.stdp import SynapsePairWiseSTDP
from src.core.learning.stdp_delayless import SynapsePairWiseSTDPWithoutDelay
from src.core.metrics.activity_metrics import ActivityMetrics
from src.core.neurons.neurons import StreamableLIFNeurons
from src.core.neurons.trace import TraceHistory
from src.core.population.pop_activity import PopulationBaseActivity
from src.core.population.pop_activity_trace import PopulationTraceHistory
from src.core.population.pop_current import PopCurrentStimulus
from src.core.population.pop_homeostasis import PopulationBaseHomeostasis
from src.helpers.network import EpisodeTracker

"""
DONE: Check the correct path for synapse_lens_selector
DONE: For population we don't need winner take all, need activity measuring instead
DONE: For population we don't need supervisor for activity
DONE: Remove delay learning for the (pos|neg) recurrent synapse connections
DONE: Add weights initializer with J/N strategies

TODO: Check the inhibitory connection (does it need to be negative, is negative w_min works fine?) 
TODO: Review Brunel Hakim source code for J setter on the neuron group

Ques: should n.A be divided by #pop and #pop_items?
Ques: Supervisor should not be for every connection, they effect dopamine!
Ques: Different Metrics must also be excluded or combine their reports!
Ques: Check the ceiling effect on homeostasis behaviour (PopHomeostasis)
Ques: Is it meaningful to reduce max weight so we get 51,49 rule of activation in the population of pos or neg?   
Ques: Should we use src.fired or src.fire_effect in the `pop-activity` -> PopulationBaseActivity
Ques: How we should handle the reinforcement for the sentences. they are really long and class based
Ques: Random connection removing in synapse.W
Ques: Double check weight initialization J/√PN
Ques: What does max_word_delay means in the words:pos connection
Ques: Should ActivitySupervisor needs to know how other pop behave or not?
Ques: Now all words are selected but we might need to trim them in `word_length_threshold`? Should we add -1 in the words layer
Ques: In neuron morphology `tau` -> max(words_spacing_gap, maximum_length_words), max_length is different from what it should be

IMPROVE: max instead of clip and slice instead of roll
IMPROVE: Resolve and fix the `self.outputs` in the reinforcement
"""


def network():
    network = Network()
    (train_df, _) = test_train_dataset(train_size=2, random_state=42)
    common_words = extract_words(train_df, word_length_threshold=10)
    words_stream = words2spikes(train_df, common_words)
    sentence_stream = sentence2spikes(train_df)
    joined_corpus = joined_corpus_maker(train_df)
    simulation_iterations = len(words_stream)

    lif_base = get_base_neuron_config()
    homeostasis_base = get_base_homeostasis()
    delay_args = get_base_delay()
    stdp_weights_args = get_weight_stdp()
    stdp_args = get_base_stdp()
    stdp_delay_args = get_base_delay_stdp()
    balanced_network_args = {"J": 5, "P": 0.5}
    population_window_size = 10

    n_episodes = 10

    # Words neuron group
    words_ng = NeuronGroup(
        net=network,
        tag="words",
        size=len(common_words),
        behaviour={
            1: StreamableLIFNeurons(
                stream=words_stream, joined_corpus=joined_corpus, **lif_base
            ),
            2: TraceHistory(max_delay=words_max_delay),
        },
    )

    # positive population neuron groups
    pop_ng_behaviours = {
        3: StreamableLIFNeurons(**lif_base),
        4: TraceHistory(max_delay=words_max_delay),
        5: PopulationTraceHistory(window_size=population_window_size),
        6: PopulationBaseActivity(window_size=population_window_size),
        7: PopulationBaseHomeostasis(**homeostasis_base),
    }
    pos_pop_ng = NeuronGroup(
        net=network,
        tag="pos",
        size=population_size,
        behaviour={
            2: PopCurrentStimulus(
                adaptive_noise_scale=0.9,
                noise_scale_factor=0.1,
                stimulus_scale_factor=1,
                synapse_lens_selector=["words:pos", 0],
            ),
            8: ActivitySupervisor(
                dopamine_decay=1 / (words_max_delay + 1),
                outputs=sentence_stream,
                class_index=1,
            ),
            10: ActivityMetrics(
                tag="pos:pop-metric",
                outputs=sentence_stream,
                class_index=1,
                episode_iterations=simulation_iterations,
            ),
            **pop_ng_behaviours,
        },
    )

    # negative population neuron groups
    neg_pop_ng = NeuronGroup(
        net=network,
        tag="neg",
        size=population_size,
        behaviour={
            2: PopCurrentStimulus(
                adaptive_noise_scale=0.9,
                noise_scale_factor=0.1,
                stimulus_scale_factor=1,
                synapse_lens_selector=["words:neg", 0],
            ),
            7: ActivitySupervisor(
                dopamine_decay=1 / (words_max_delay + 1),
                outputs=sentence_stream,
                class_index=0,
            ),
            9: ActivityMetrics(
                tag="neg:pop-metric",
                outputs=sentence_stream,
                class_index=0,
                episode_iterations=simulation_iterations,
            ),
            **pop_ng_behaviours,
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
            8: SynapsePairWiseSTDP(**stdp_args, **stdp_delay_args, **stdp_weights_args),
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
            8: SynapsePairWiseSTDP(**stdp_args, **stdp_delay_args, **stdp_weights_args),
        },
    )
    # pos_pop_ng -> pos_pop_ng
    SynapseGroup(
        net=network,
        src=pos_pop_ng,
        dst=pos_pop_ng,
        tag="pos:pos",
        behaviour={
            8: SynapsePairWiseSTDPWithoutDelay(
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
            8: SynapsePairWiseSTDPWithoutDelay(
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
            8: SynapsePairWiseSTDPWithoutDelay(
                **stdp_args, w_min=-stdp_weights_args["w_max"], w_max=0
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
            8: SynapsePairWiseSTDPWithoutDelay(
                **stdp_args, w_min=-stdp_weights_args["w_max"], w_max=0
            ),
        },
    )

    network.initialize(info=False)

    """ TRAINING """
    for _ in tqdm(range(n_episodes), "Training..."):
        EpisodeTracker.update()
        network.iteration = 0
        network.simulate_iterations(simulation_iterations, measure_block_time=False)
        for metric_tag in ["neg:pop-metric", "pos:pop-metric"]:
            network[metric_tag, 0].reset()


if __name__ == "__main__":
    network()
