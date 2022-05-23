from src.core.visualizer.history_recoreder_1d import HistoryRecorder1D
from src.core.visualizer.history_recoreder_2d import HistoryRecorder2D

dopamine_plotter = HistoryRecorder1D(title="dopamine", window_size=25, enabled=False)
dw_plotter = HistoryRecorder2D(title="dw", window_size=4, enabled=False)
w_plotter = HistoryRecorder2D(title="w", window_size=25, enabled=False)
delay_plotter = HistoryRecorder2D(title="delay", window_size=25, enabled=False)
selected_delay_plotter = HistoryRecorder1D(
    title="selected - delay", window_size=25, enabled=True
)
selected_dw_plotter = HistoryRecorder1D(
    title="selected - dw", window_size=25, enabled=False
)
threshold_plotter = HistoryRecorder1D(title="threshold", enabled=False)
activity_plotter = HistoryRecorder1D(title="activity", window_size=25, enabled=False)
words_stimulus_plotter = HistoryRecorder1D(
    title="words_stimulus", window_size=25, enabled=False
)
