#!/usr/bin/env python3
"""
This module contains a function that creates
a pandas DataFrame from a NumPy array.
"""

import pandas as pd


def from_numpy(array):
    """
    Creates a pd.DataFrame from a np.ndarray.

    Args:
        array: The np.ndarray used to create the DataFrame.

    Returns:
        The newly created pd.DataFrame.
    """
    num_cols = array.shape[1]
    col_names = [chr(65 + i) for i in range(num_cols)]
    return pd.DataFrame(array, columns=col_names)
