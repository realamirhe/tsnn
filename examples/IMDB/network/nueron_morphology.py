from examples.IMDB.config import words_spacing_gap


def get_base_neuron_config(**kargs):
    lif_base = {
        "v_rest": -65,
        "v_reset": -65,
        "threshold": -55,
        "dt": 1.0,
        "R": 1,
        "tau": max(words_spacing_gap, 1),  # 2
        **kargs,
    }
    return lif_base


def get_base_homeostasis(**kargs):
    homeostasis_window_size = kargs.get("homeostasis_window_size", 100)
    homeostasis_base = {
        "window_size": homeostasis_window_size,
        "updating_rate": 0.01,
        "population_count": 2,
        # activity must occur every 3 iterations, so we need over each window slide
        "activity_rate": homeostasis_window_size * 0.3,
    }
    return homeostasis_base


if __name__ == "__main__":
    print(get_base_neuron_config())
