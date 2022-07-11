from examples.IMDB.config import words_spacing_gap


def get_base_neuron_config(**kwargs):
    lif_base = {
        "v_rest": -65,
        "v_reset": -65,
        "threshold": -60,
        "dt": 1.0,
        "R": 1,
        "tau": max(words_spacing_gap, 1),  # 2
        **kwargs,
    }
    return lif_base


def get_base_homeostasis(**kwargs):
    homeostasis_window_size = kwargs.get("homeostasis_window_size", 100)
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
