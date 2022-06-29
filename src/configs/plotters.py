from src.configs.feature_flags import enable_plotter
from src.core.visualizer.history_recoreder_1d import HistoryRecorder1D
from src.core.visualizer.history_recoreder_2d import HistoryRecorder2D

dopamine_plotter = HistoryRecorder1D(
    title="dopamine", window_size=25, enabled=enable_plotter and False
)
dw_plotter = HistoryRecorder2D(
    title="dw", window_size=4, enabled=enable_plotter and False
)
w_plotter = HistoryRecorder2D(
    title="w", window_size=25, enabled=enable_plotter and False
)
delay_plotter = HistoryRecorder2D(
    title="delay", window_size=25, enabled=enable_plotter and False
)
activity_plotter = HistoryRecorder1D(
    title="activity",
    window_size=25,
    vertical_history_separator=True,
    enabled=enable_plotter and False,
    should_copy_on_add=True,
)
selected_dw_plotter = HistoryRecorder1D(
    title="selected::dw",
    vertical_history_separator=True,
    window_size=25,
    enabled=enable_plotter and False,
)
threshold_plotter = HistoryRecorder1D(
    title="threshold",
    vertical_history_separator=True,
    should_copy_on_add=True,
    enabled=enable_plotter and True,
    every_n_episode=5,
)
words_stimulus_plotter = HistoryRecorder1D(
    title="words_stimulus",
    window_size=25,
    should_copy_on_add=True,
    enabled=enable_plotter and False,
)
selected_delay_plotter = HistoryRecorder1D(
    title="selected::delay",
    window_size=25,
    vertical_history_separator=True,
    enabled=enable_plotter and True,
    every_n_episode=5,
)
selected_weights_plotter = HistoryRecorder1D(
    title="selected::weights",
    window_size=3,
    vertical_history_separator=True,
    enabled=enable_plotter and True,
    every_n_episode=5,
    save_as_csv=False,
)

dst_firing_plotter = HistoryRecorder1D(
    title="dst firing",
    vertical_history_separator=True,
    should_copy_on_add=True,
    save_as_csv=False,
    enabled=enable_plotter and False,
)
