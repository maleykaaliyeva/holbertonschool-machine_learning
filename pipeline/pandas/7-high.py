#!/usr/bin/env python3
"""
Sorts a DataFrame by the High column in descending order.
"""


def high(df):
    """
    Sorts the DataFrame by High price in descending order.

    Args:
        df: A pd.DataFrame.

    Returns:
        The sorted pd.DataFrame.
    """
    return df.sort_values(by="High", ascending=False)
