#!/usr/bin/env python3
"""
Computes descriptive statistics for all columns
except the Timestamp column.
"""


def analyze(df):
    """
    Computes descriptive statistics for all columns except Timestamp.

    Args:
        df: A pd.DataFrame.

    Returns:
        A new pd.DataFrame containing descriptive statistics.
    """
    return df.drop(columns=["Timestamp"]).describe()
