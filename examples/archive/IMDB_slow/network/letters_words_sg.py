import numpy as np

from PymoNNto import SynapseGroup
from examples.archive.IMDB_slow.configurations.network_config import (
    max_delay,
    average_length,
)
from examples.archive.IMDB_slow.configurations.neurons_config import lif_base_config
from src.core.learning.delay import SynapseDelay
from src.core.learning.stdp import SynapsePairWiseSTDP


def letters_words_sg(network, src, dst):
    lif_base = lif_base_config()
    return SynapseGroup(
        net=network,
        src=src,
        dst=dst,
        tag="GLUTAMATE",
        behaviour={
            1: SynapseDelay(
                tag="delay",
                max_delay=max_delay,
                mode="random",
                use_shared_weights=False,
            ),
            8: SynapsePairWiseSTDP(
                tag="stdp",
                tau_plus=4.0,
                tau_minus=4.0,
                a_plus=0.2,  # 0.02
                a_minus=-0.1,  # 0.01
                delay_a_plus=0.2,
                delay_a_minus=-0.5,
                dt=1.0,
                w_min=0,
                # epsilon: delay epsilon increase update, reduce full stimulus by tiny amount
                w_max=np.round(
                    (lif_base["threshold"] - lif_base["v_rest"]) / average_length + 0.7,
                    decimals=1,
                ),
                min_delay_threshold=1,
                weight_update_strategy=None,
                stdp_factor=0.02,
                max_delay=max_delay,
                delay_factor=0.02,  # episode increase
            ),
        },
    )
