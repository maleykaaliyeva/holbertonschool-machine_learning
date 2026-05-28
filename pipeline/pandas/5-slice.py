#!/usr/bin/env python3
"""
Extracts selected columns from a DataFrame
and selects every 60th row.
"""


def slice(df):
    """
    Extracts High, Low, Close, and Volume_(BTC) columns,
    then selects every 60th row.

    Args:
        df: A pd.DataFrame.

    Returns:
        The sliced pd.DataFrame.
    """
    return df[["High", "Low", "Close", "Volume_(BTC)"]].iloc[::60]
