"""
visualise.py

Plotting helpers for sanity-checking data before it feeds into the solver.
Currently just the rainfall/level comparison plot used to confirm the
Haw Bridge + Trimpley data actually reproduces the shape of the real 2007
flood event (rainfall peak followed by a lagged level rise).
"""

import matplotlib.pyplot as plt
import pandas as pd


def plot_rainfall_and_level(rain_df: pd.DataFrame, level_df: pd.DataFrame) -> None:
    """
    Plot rainfall and river level on two stacked, time-aligned subplots.

    Args:
        rain_df: output of load_measure() for a rainfall gauge — expects
            'dateTime' and 'value' columns.
        level_df: output of load_measure() for a level gauge — same
            column expectations as rain_df.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(rain_df['dateTime'], rain_df['value'])
    axes[0].set_ylabel('Rainfall (mm)')

    axes[1].plot(level_df['dateTime'], level_df['value'])
    axes[1].set_ylabel('Level (m)')

    plt.xlabel('Date')
    plt.tight_layout()
    plt.show()
