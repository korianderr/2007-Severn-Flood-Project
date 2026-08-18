"""
pot_analysis.py

Peaks-over-threshold (GPD) analysis of Haw Bridge river level, as an
alternative to the annual-maxima GEV fit in extreme_value.py. This fit
on 52 data points was destabilised by an outlier (2007). POT uses every
independent flood peak in the full daily record instead of one day per
year.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_ingestion import stage_to_aod


def load_level_series(csv_path: str) -> pd.DataFrame:
    """
    Load the full daily Haw Bridge level record, keeping only
    Good readings, converted from local stage datum (mASD) to mAOD.
 
    Returns:
        pd.DataFrame with columns 'date' (Timestamp) and 'level_mAOD'
        (float), Good-quality readings only, sorted by date.
    """
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
 
    good = df[df['quality'] == 'Good'].copy()
    good['level_mAOD'] = good['value'].apply(stage_to_aod)
 
    return good[['date', 'level_mAOD']].sort_values('date').reset_index(drop=True)

def mean_residual_life(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Compute the mean excess above each candidate threshold: the average
    amount by which readings exceed that threshold, among readings that
    do exceed it.

    Returns:
        array the same length as thresholds - mean excess at each one.
        NaN where fewer than 2 readings exceed that threshold (too few
        to average meaningfully).
    """
    mean_excess = np.full(len(thresholds), np.nan)
 
    for i, u in enumerate(thresholds):
        exceedances = values[values > u] - u
        if len(exceedances) > 1:
            mean_excess[i] = exceedances.mean()
 
    return mean_excess


def plot_mean_residual_life(values: np.ndarray, thresholds: np.ndarray) -> None:
    """
    Plot threshold vs. mean excess, to visually identify where the
    relationship becomes roughly linear.
    """
    mean_excess = mean_residual_life(values, thresholds)
 
    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, mean_excess, marker='o', markersize=3)
    plt.xlabel('Threshold, mAOD')
    plt.ylabel('Mean excess above threshold, m')
    plt.title('Mean residual life plot — Haw Bridge river level')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    level_series = load_level_series('data/Haw-Bridge-level-daily-Qualified.csv')
 
    print(f"Total readings: {len(level_series)}")
    print(f"Date range: {level_series['date'].min().date()} to "
          f"{level_series['date'].max().date()}")
    print(f"Level range: {level_series['level_mAOD'].min():.2f} to "
          f"{level_series['level_mAOD'].max():.2f} mAOD")
 
    # Candidate thresholds: from roughly the median up to just below the max
    values = level_series['level_mAOD'].values
    candidate_thresholds = np.linspace(
        np.median(values), np.percentile(values, 99.5), 100
    )
 
    plot_mean_residual_life(values, candidate_thresholds)
 
