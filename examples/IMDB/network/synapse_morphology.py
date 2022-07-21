from examples.IMDB.config import words_max_delay
from examples.IMDB.network.nueron_morphology import get_base_neuron_config


def get_base_delay():
    return {
        "max_delay": words_max_delay,
        "mode": "random",
        "use_shared_weights": False,
    }


def get_base_stdp(**kwargs):
    return {
        "a_plus": 0.1,  # 0.02
        "a_minus": -0.1,  # 0.01
        "dt": 1.0,
        "weight_update_strategy": "soft-bound",
        "stdp_factor": 0.02,
        **kwargs,
    }


def get_base_delay_stdp():
    return {
        "delay_a_plus": 0.2,
        "delay_a_minus": -0.5,
        "delay_factor": 0.02,  # episode increase
        "max_delay": words_max_delay,
        "min_delay_threshold": 1,
    }


def get_weight_stdp():
    lif_base = get_base_neuron_config()
    return {
        "w_min": 0,
        "w_max": 7
        # "w_max": np.round(
        #     (lif_base["threshold"] - lif_base["v_rest"]) / average_words_length + 0.7,
        #     decimals=1,
        # ),
    }