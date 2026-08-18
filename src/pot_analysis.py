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
from scipy.stats import genpareto
from pathlib import Path

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


def decluster_exceedances(
    level_series: pd.DataFrame,
    threshold: float,
    min_gap_days: int = 7,
) -> pd.DataFrame:
    """
    Extract independent flood events from the exceedances above a
    threshold, collapsing runs of consecutive/nearby exceedance days
    down to a single peak per event.
 
    Method: sort exceedance days chronologically, compute the gap
    between each one and the previous exceedance day. A gap strictly
    greater than min_gap_days starts a new event. Within each resulting
    group, keep only the single highest reading.
 
    Args:
        level_series: output of load_level_series() - columns 'date'
            and 'level_mAOD'.
        threshold: the GPD threshold (mAOD).
        min_gap_days: minimum number of days the river must spend below
            threshold before the next exceedance counts as a new,
            independent event, rather than the tail end of the same
            flood..
 
    Returns:
        pd.DataFrame with columns 'date' and 'level_mAOD' - one row per
        independent event, holding that event's peak reading and the
        date it occurred on.
    """
    exceedances = (
        level_series[level_series['level_mAOD'] > threshold]
        .sort_values('date')
        .reset_index(drop=True)
    )
 
    if exceedances.empty:
        return exceedances
 
    gap_days = exceedances['date'].diff().dt.days
    starts_new_event = gap_days.isna() | (gap_days > min_gap_days)
    event_id = starts_new_event.cumsum()
 
    peak_idx = exceedances.groupby(event_id)['level_mAOD'].idxmax()
    events = exceedances.loc[peak_idx, ['date', 'level_mAOD']]
 
    return events.reset_index(drop=True)


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


def fit_gpd(events: pd.DataFrame, threshold: float) -> tuple[float, float]:
    """
    Fit a GPD to declustered event peaks.

    Args:
        events: output of decluster_exceedances() - columns 'date' and
            'level_mAOD'.
        threshold: the same threshold used to produce events (mAOD).
 
    Returns:
        (c, scale) - GPD shape and scale.
    """
    excesses = events['level_mAOD'].values - threshold
    c, loc, scale = genpareto.fit(excesses, floc=0)
    return c, scale


def gpd_parameter_stability(
    level_series: pd.DataFrame,
    thresholds: np.ndarray,
    min_gap_days: int = 7,
    min_events: int = 10,
) -> pd.DataFrame:
    """
    Fit a GPD at each candidate threshold and record the diagnostics used
    to pick a defensible one.

    The GPD is only asymptotically justified above some unknown threshold.
    Above it, the shape xi is constant and the scale varies with threshold
    in a known way: sigma_u = sigma_u0 + xi * (u - u0). Subtracting that
    dependence gives the modified scale, sigma* = sigma_u - xi * u, which
    is constant in u if and only if the GPD model actually holds above u.
    So xi and sigma* both flattening is the signal.

    Args:
        level_series: output of load_level_series() - columns 'date' and
            'level_mAOD'.
        thresholds: candidate thresholds (mAOD) to test.
        min_gap_days: passed through to decluster_exceedances().
        min_events: skip thresholds with fewer independent events than
            this - the fit is too unstable to be worth plotting.

    Returns:
        pd.DataFrame, one row per usable threshold, columns:
        'threshold', 'n_events', 'events_per_year', 'shape', 'scale',
        'mod_scale', 'upper_bound' (NaN where the tail is unbounded).
    """
    record_years = (
        level_series['date'].max() - level_series['date'].min()
    ).days / 365.25

    records = []
    for u in thresholds:
        events = decluster_exceedances(level_series, u, min_gap_days)
        if len(events) < min_events:
            continue

        c, scale = fit_gpd(events, u)

        # Bounded above only when xi < 0; otherwise the tail is infinite.
        upper_bound = u - scale / c if c < 0 else np.nan

        records.append({
            'threshold': u,
            'n_events': len(events),
            'events_per_year': len(events) / record_years,
            'shape': c,
            'scale': scale,
            'mod_scale': scale - c * u,
            'upper_bound': upper_bound,
        })

    return pd.DataFrame.from_records(records)


def plot_parameter_stability(stability: pd.DataFrame) -> None:
    """
    Plot shape, modified scale, and events/year against threshold.

    Pick the lowest threshold where shape and modified scale both flatten
    out, then sanity-check that events/year lands somewhere plausible for
    genuine flood peaks (roughly 1-3, not 4+).
    """
    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)

    axes[0].plot(stability['threshold'], stability['shape'], marker='o', markersize=3)
    axes[0].axhline(0, color='grey', linewidth=0.8, linestyle='--')
    axes[0].set_ylabel('Shape (xi)')

    axes[1].plot(stability['threshold'], stability['mod_scale'], marker='o', markersize=3)
    axes[1].set_ylabel('Modified scale')

    axes[2].plot(stability['threshold'], stability['events_per_year'], marker='o', markersize=3)
    axes[2].axhspan(1, 3, color='C2', alpha=0.15)
    axes[2].set_ylabel('Independent events / year')
    axes[2].set_xlabel('Threshold, mAOD')

    axes[0].set_title('GPD parameter stability — Haw Bridge river level')
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
     # Threshold chosen from the straight-looking stretch of the plot
    # (roughly 9.0-10.5 mAOD).
    stability_thresholds = np.linspace(9.0, 11.2, 30)
    stability = gpd_parameter_stability(level_series, stability_thresholds)
    print(stability.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    plot_parameter_stability(stability)
    events = decluster_exceedances(level_series, stability_thresholds, min_gap_days=7)

    print(f"\nExceedance days above {stability_thresholds} mAOD: "
          f"{(level_series['level_mAOD'] > stability_thresholds).sum()}")
    print(f"Independent events after declustering: {len(events)}")

    # Sanity check: 2007's flood should collapse to one event.
    events_2007 = events[events['date'].dt.year == 2007]
    print(f"\nEvents in 2007:\n{events_2007}")

    c, scale = fit_gpd(events, stability_thresholds)
    print(f"\nGPD fit: c (shape, xi) = {c:.3f}, scale = {scale:.3f}")
    boundary = 9.5 - scale / c
    print(f"\nBoundary: {boundary:.3f}mAOD")
