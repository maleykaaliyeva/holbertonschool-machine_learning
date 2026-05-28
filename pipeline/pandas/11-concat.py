#!/usr/bin/env python3
"""
Concatenates selected Bitstamp data with Coinbase data
after setting Timestamp as the index.
"""

import pandas as pd

index = __import__('10-index').index


def concat(df1, df2):
    """
    Indexes both DataFrames on Timestamp, selects Bitstamp rows
    up to timestamp 1417411920, and concatenates them with Coinbase.

    Args:
        df1: Coinbase pd.DataFrame.
        df2: Bitstamp pd.DataFrame.

    Returns:
        The concatenated pd.DataFrame.
    """
    df1 = index(df1)
    df2 = index(df2)

    df2 = df2.loc[:1417411920]

    return pd.concat([df2, df1], keys=["bitstamp", "coinbase"])
