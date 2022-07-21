def get_base_neuron_config(**kwargs):
    lif_base = {
        "v_rest": -65,
        "v_reset": -65,
        "threshold": -60,
        "dt": 1.0,
        "R": 1,
        "tau": 60,  # how long it takes for words to be forgotten (|sentence| / 2)
        **kwargs,
    }
    return lif_base


def get_base_homeostasis(**kwargs):
    homeostasis_base = {
        "updating_rate": 0.07,  # 1 / HomeostasisEnvironment.num_sentences,
        **kwargs,
    }
    return homeostasis_base


if __name__ == "__main__":
    print(get_base_neuron_config())
