#!/usr/bin/env python3
"""
Concatenates selected Bitstamp and Coinbase data
using a hierarchical MultiIndex.
"""

import pandas as pd

index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Indexes both DataFrames on Timestamp, selects rows from
    1417411980 to 1417417980 inclusive, concatenates them,
    and rearranges the MultiIndex.

    Args:
        df1: Coinbase pd.DataFrame.
        df2: Bitstamp pd.DataFrame.

    Returns:
        The concatenated pd.DataFrame.
    """
    df1 = index(df1)
    df2 = index(df2)

    df1 = df1.loc[1417411980:1417417980]
    df2 = df2.loc[1417411980:1417417980]

    df = pd.concat([df2, df1], keys=["bitstamp", "coinbase"])

    return df.swaplevel(0, 1).sort_index()
