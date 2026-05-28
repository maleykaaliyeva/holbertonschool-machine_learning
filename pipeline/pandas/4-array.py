#!/usr/bin/env python3
"""
Selects the last 10 rows of High and Close columns
and converts them to a NumPy array.
"""


def array(df):
    """
    Selects the last 10 rows of High and Close columns.

    Args:
        df: A pd.DataFrame containing High and Close columns.

    Returns:
        A numpy.ndarray with the selected values.
    """
    return df[["High", "Close"]].tail(10).to_numpy()
