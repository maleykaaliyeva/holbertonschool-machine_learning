#!/usr/bin/env python3
"""Preprocessing module for BTC time series forecasting data."""
import pandas as pd
import numpy as np


def preprocess_data(file_path):
    """Preprocesses BTC raw dataset into hourly aggregated windows.

    Args:
        file_path (str): Path to raw CSV file.

    Returns:
        pd.DataFrame: Processed hourly data.
    """
    df = pd.read_csv(file_path)

    # Convert Timestamp from Unix seconds to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df.set_index('Timestamp', inplace=True)

    # Handle missing values by forward filling, then backward filling
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Resample 1-minute data into 1-hour windows
    hourly_df = df.resample('1h').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume_(BTC)': 'sum',
        'Volume_(Currency)': 'sum',
        'Weighted_Price': 'mean'
    }).dropna()

    return hourly_df


if __name__ == '__main__':
    pass
