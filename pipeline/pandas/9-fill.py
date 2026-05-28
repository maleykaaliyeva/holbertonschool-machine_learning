#!/usr/bin/env python3
"""
Modifies a DataFrame by removing Weighted_Price
and filling missing values.
"""


def fill(df):
    """
    Removes Weighted_Price and fills missing values.

    Args:
        df: A pd.DataFrame.

    Returns:
        The modified pd.DataFrame.
    """
    df = df.drop(columns=["Weighted_Price"])

    df["Close"] = df["Close"].ffill()

    df["High"] = df["High"].fillna(df["Close"])
    df["Low"] = df["Low"].fillna(df["Close"])
    df["Open"] = df["Open"].fillna(df["Close"])

    df["Volume_(BTC)"] = df["Volume_(BTC)"].fillna(0)
    df["Volume_(Currency)"] = df["Volume_(Currency)"].fillna(0)

    return df
