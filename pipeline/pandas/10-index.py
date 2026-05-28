#!/usr/bin/env python3
"""
Sets the Timestamp column as the index of a DataFrame.
"""


def index(df):
    """
    Sets the Timestamp column as the index.

    Args:
        df: A pd.DataFrame.

    Returns:
        The modified pd.DataFrame.
    """
    return df.set_index("Timestamp")
