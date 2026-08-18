"""
extreme_value.py
 
Extreme value analysis for the risk model: fitting a distribution to
historical rainfall extremes so we can Monte Carlo sample synthetic storm
scenarios beyond what's in the historical record. Starts with annual maxima
extraction (GEV fit); the GEV fitting itself and Monte Carlo sampling will
be added here as they're built.
"""
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genextreme, kstest

from data_ingestion import stage_to_aod

def load_annual_max_with_date(csv_path: str) -> pd.DataFrame:
    """
    Load a raw multi-year EA hydrology rainfall CSV and extract one annual
    maximum per year, keeping only Good readings.
    
    Suspect/Missing/Unchecked readings are excluded rather than just
    Missing, since a wrongly-high Suspect reading landing on a year's
    maximum would distort that year's extreme value going into the
    GEV fit. 

    Also keeps the date each year's maximum occurred on. Needed for
    event-matching against another station's data.
 
    Returns:
        pd.DataFrame indexed by year (int), columns 'date' (Timestamp)
        and 'value' (float) — the date and value of that year's maximum
        among Good-quality readings only.
    """
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
 
    good = df[df['quality'] == 'Good']
    peak_idx = good.groupby(good['date'].dt.year)['value'].idxmax()
 
    peaks = good.loc[peak_idx, ['date', 'value']].copy()
    peaks.index = peaks['date'].dt.year
    peaks.index.name = 'year'
 
    return peaks


def build_event_matched_pairs(
    rainfall_csv_path: str,
    level_csv_path: str,
    lag_days: int = 3,
) -> pd.DataFrame:
    """
    Pair each year's rainfall peak with the peak river level in the days
    immediately following it.
 
    Uses the ~1-day rainfall-to-level lag: for each year, finds the exact
    date of the rainfall peak, then looks at level readings in a window
    from that date to lag_days days after it, and takes the peak level
    within that window.
 
    Args:
        rainfall_csv_path: path to the raw rainfall CSV (date, value,
            quality columns).
        level_csv_path: path to the raw level CSV (date, value, quality
            columns) — in local stage datum (mASD); converted to mAOD
            internally via stage_to_aod().
        lag_days: how many days after the rainfall peak to search for the
            level response. Default 3 — generous around the ~1-day lag
            seen in Week 1, without stretching so far that an unrelated
            later event could get pulled in.
 
    Returns:
        pd.DataFrame indexed by year, columns 'rainfall_date',
        'rainfall_mm', 'level_mAOD'. A year is dropped if no Good-quality
        level reading exists anywhere in its lag window.
    """
    rainfall_peaks = load_annual_max_with_date(rainfall_csv_path)
 
    level_df = pd.read_csv(level_csv_path)
    level_df['date'] = pd.to_datetime(level_df['date'])
    level_good = level_df[level_df['quality'] == 'Good']
 
    records = []
    for year, row in rainfall_peaks.iterrows():
        window_start = row['date']
        window_end = row['date'] + pd.Timedelta(days=lag_days)
 
        window = level_good[
            (level_good['date'] >= window_start) & (level_good['date'] <= window_end)
        ]
 
        if window.empty:
            continue
 
        peak_level_stage = window['value'].max()
        records.append({
            'year': year,
            'rainfall_date': row['date'],
            'rainfall_mm': row['value'],
            'level_mAOD': stage_to_aod(peak_level_stage),
        })
 
    return pd.DataFrame.from_records(records, index='year')


def plot_rainfall_vs_level(paired: pd.DataFrame) -> None:
    """
    Scatter plot of event-matched rainfall peak vs. the river level peak
    that followed it, to check whether a real relationship is visible
    before fitting anything to it.
    """

    plt.figure(figsize=(6, 5))
    plt.scatter(paired['rainfall_mm'], paired['level_mAOD'])
    plt.xlabel('Rainfall peak (mm)')
    plt.ylabel('River level peak, mAOD (within lag window after rainfall peak)')
    plt.title('Event-matched rainfall vs. river level peaks')
    plt.tight_layout()
    plt.show()


def plot_return_levels(annual_max, params):
    """
    Plot return periods at each rainfall reading. Return period represents expected
    frequency of seeing that rainfall.

    Plot empirical vs. fitted return levels, to visually sanity-check a
    GEV fit before trusting it enough to sample synthetic scenarios from.

    Returns: Plot of return period against rainfall value.
    """
    sorted_maxima = annual_max.sort_values().values

    n = len(sorted_maxima)
    T = np.zeros(n)

    for i in range(1, n + 1):
        p_i = i / (n + 1)
        T_i = 1 / (1 - p_i)
        T[i - 1] = T_i

    c, loc, scale = params
    T_smooth = np.logspace(np.log10(1.1), np.log10(200), 100)
    fitted_values = genextreme.ppf(1 - 1 / T_smooth, c, loc, scale)

    plt.scatter(T, sorted_maxima, label='Empirical (observed years)')
    plt.plot(T_smooth, fitted_values, color='C1', label='Fitted GEV')
    plt.xscale('log')
    plt.xlabel('Return period (years)')
    plt.ylabel('Annual maximum river level (mAOD)')
    plt.title('Return level plot — Haw Bridge river level')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_return_levels_compare(annual_max: pd.Series, params_by_label: dict) -> None:
    """
    Same idea as plot_return_levels(), but overlays multiple fitted
    curves against one empirical scatter. Built to directly compare
    MLE vs L-moments and see which tracks the data better.
 
    Args:
        annual_max: output of load_annual_maxima().
        params_by_label: e.g. {'MLE': params_mle, 'L-moments': params_lmom}.
    """
    sorted_maxima = annual_max.sort_values().values
    n = len(sorted_maxima)
 
    T = np.zeros(n)
    for i in range(1, n + 1):
        T[i - 1] = 1 / (1 - i / (n + 1))
 
    T_smooth = np.logspace(np.log10(1.1), np.log10(200), 100)
 
    plt.figure(figsize=(7, 5))
    plt.scatter(T, sorted_maxima, label='Empirical (observed years)', zorder=3)
    for label, (c, loc, scale) in params_by_label.items():
        fitted_values = genextreme.ppf(1 - 1 / T_smooth, c, loc, scale)
        plt.plot(T_smooth, fitted_values, label=f'Fitted GEV ({label})')
    plt.xscale('log')
    plt.xlabel('Return period (years)')
    plt.ylabel('Annual maximum river level (mAOD)')
    plt.title('Return level plot — Haw Bridge, MLE vs L-moments')
    plt.legend()
    plt.tight_layout()
    plt.show()


### GEV FIT
def fit_gev(annual_max: pd.Series) -> tuple[float, float, float]:
    return genextreme.fit(annual_max.values)


def fit_gev_lmoments(annual_max: pd.Series) -> tuple[float, float, float]:
    """
    Fit GEV via L-moments.
 
    MLE shape estimates are unstable with ~50 years of data,
    especially when one extreme dominates a record (e.g. Haw
    Bridge's 2007 reading). MLE can explain that outlier by placing a
    bounding ceiling almost on top of it, fitting the outlier at the
    cost of badly misfitting everything else. L-moments weights the
    whole ordered sample rather than chasing the single most extreme
    point, so it doesn't fall into that trap.
    """
    from lmoments3 import distr
    fitted = distr.gev.lmom_fit(annual_max.values)
    return fitted['c'], fitted['loc'], fitted['scale']


def gof_test_gev(annual_max: pd.Series, params: tuple[float, float, float]) -> tuple[float, float]:
    """
    Goodness-of-fit test: how consistent is the annual maxima series
    with having come from the fitted GEV distribution?
 
    Returns:
        (statistic, p_value). 
    """
    c, loc, scale = params
    return kstest(annual_max.values, genextreme.cdf, args=(c, loc, scale))


if __name__ == "__main__":
    annual_max = load_annual_max_with_date('data/Trimpley-rainfall-daily-Qualified.csv')['value']
    params = fit_gev(annual_max) # c (shape), loc (location), scale 

    # Try river level
    level_annual_max = load_annual_max_with_date('data/Haw-Bridge-level-daily-Qualified.csv')['value']
    level_annual_max_aod = level_annual_max.apply(stage_to_aod)
    level_params = fit_gev(level_annual_max_aod)

    print(f"Years with a valid annual maximum: {len(level_annual_max_aod)}")
    print(f"Year range: {level_annual_max_aod.index.min()}-{level_annual_max_aod.index.max()}")
    print(f"Largest annual maximum: {level_annual_max_aod.max():.1f} mm "
          f"(year {level_annual_max_aod.idxmax()})")
    print(f"Smallest annual maximum: {level_annual_max_aod.min():.1f} mm "
          f"(year {level_annual_max_aod.idxmin()})")
    expected_years = set(range(level_annual_max_aod.index.min(), level_annual_max_aod.index.max() + 1))
    missing_years = sorted(expected_years - set(level_annual_max_aod.index))
    print(f"Years with no valid annual maximum: {missing_years}")

    statistic, p_value = gof_test_gev(level_annual_max_aod, level_params)
    print(f"Goodness of fit test: statistic={statistic:.3f}, p-value={p_value:.3f}")
    plot_return_levels(level_annual_max_aod, level_params)

    level_params_mle = fit_gev(level_annual_max_aod)
    level_params_lmom = fit_gev_lmoments(level_annual_max_aod)

    print("MLE:", level_params_mle)
    print("L-moments:", level_params_lmom)

    plot_return_levels_compare(level_annual_max_aod, {
        'MLE': level_params_mle,
        'L-moments': level_params_lmom,
    })

    # Event-matched rainfall -> level pairing
    paired = build_event_matched_pairs(
        'data/Trimpley-rainfall-daily-Qualified.csv',
        'data/Haw-Bridge-level-daily-Qualified.csv',
        lag_days=3,
    )
    print(f"Event-matched pairs: {len(paired)} "
          f"(out of {len(annual_max)} rainfall years)")
    plot_rainfall_vs_level(paired)
