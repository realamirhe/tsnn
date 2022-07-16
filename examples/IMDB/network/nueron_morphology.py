def get_base_neuron_config(**kwargs):
    lif_base = {
        "v_rest": -65,
        "v_reset": -65,
        "threshold": -45,
        "dt": 1.0,
        "R": 1,
        "tau": 60,  # how long it takes for words to be forgotten (|sentence| / 2)
        **kwargs,
    }
    return lif_base


def get_base_homeostasis(**kwargs):
    homeostasis_window_size = kwargs.get("homeostasis_window_size", 100)
    homeostasis_base = {
        "window_size": homeostasis_window_size,
        "updating_rate": 0.007,
        # "population_count": 2,
        # neuron base non-pop
        # activity must occur every 3 iterations, so we need over each window slide
        "activity_rate": 51 * 120,
    }
    return homeostasis_base


if __name__ == "__main__":
    print(get_base_neuron_config())
