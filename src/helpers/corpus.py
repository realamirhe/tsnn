import numpy as np
import pandas as pd


def replicate_df_rows(corpus_dataframe, replication_count=2):
    return pd.DataFrame(
        np.repeat(corpus_dataframe.values, replication_count, axis=0),
        columns=corpus_dataframe.columns,
    )
