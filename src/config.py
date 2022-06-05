from collections import namedtuple

#  Network Feature flags
feature_flags = dict(
    enable_magic_delays=False,
    enable_magic_weights=True,
    enable_delay_update_in_stdp=True,
    enable_cm_plot=True,
    enable_metric_logs=True,
)
feature_flags = namedtuple("FeatureFlags", feature_flags.keys())(**feature_flags)
