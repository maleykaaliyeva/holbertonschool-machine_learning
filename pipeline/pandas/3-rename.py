#!/usr/bin/env python3
"""
Renames the Timestamp column, converts it to datetime,
and returns only the Datetime and Close columns.
"""

import pandas as pd


def rename(df):
    """
    Renames Timestamp to Datetime and converts it to datetime format.

    Args:
        df: A pd.DataFrame containing Timestamp and Close columns.

    Returns:
        The modified pd.DataFrame with only Datetime and Close columns.
    """
    df = df.rename(columns={"Timestamp": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit="s")
    return df[["Datetime", "Close"]]
