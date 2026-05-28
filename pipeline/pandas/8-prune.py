#!/usr/bin/env python3
"""
Removes rows where the Close column has NaN values.
"""


def prune(df):
    """
    Removes entries where Close has NaN values.

    Args:
        df: A pd.DataFrame.

    Returns:
        The modified pd.DataFrame.
    """
    return df.dropna(subset=["Close"])
