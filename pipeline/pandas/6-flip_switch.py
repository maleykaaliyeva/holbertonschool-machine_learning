#!/usr/bin/env python3
"""
Sorts a DataFrame in reverse chronological order
and transposes it.
"""


def flip_switch(df):
    """
    Sorts the DataFrame by Timestamp in descending order,
    then transposes the DataFrame.

    Args:
        df: A pd.DataFrame.

    Returns:
        The transformed pd.DataFrame.
    """
    df = df.sort_values(by="Timestamp", ascending=False)
    return df.transpose()
