import string
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

# NLP & Corpus configuration
corpus_config = dict(
    words_spacing_gap=7,
    language=string.ascii_lowercase + " ",
    letters=string.ascii_lowercase,
    words=["abc", "omn"],
)
corpus_config = namedtuple("Corpus", corpus_config.keys())(**corpus_config)

# Network configuration
network_config = dict(
    is_debug_mode=True,
)
network_config = namedtuple("Network", network_config.keys())(**network_config)
