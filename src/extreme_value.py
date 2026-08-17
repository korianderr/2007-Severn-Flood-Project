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
 
def load_annual_maxima(csv_path: str) -> pd.Series:
    """
    Load a raw multi-year EA hydrology rainfall CSV and extract one annual
    maximum per year, keeping only 'Good'-quality readings.
 
    Suspect/Missing/Unchecked readings are excluded rather than just
    Missing, since a wrongly-high Suspect reading landing on a year's
    maximum would distort that year's extreme value going into the
    GEV fit. 
 
    Returns:
        pd.Series indexed by year (int), values = that year's maximum daily
        rainfall (mm) among Good-quality readings only. Years with zero
        Good readings will simply be absent from the index — check the
        returned series' length against the expected number of years.
    """
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
 
    good = df[df['quality'] == 'Good']
    annual_max = good.groupby(good['date'].dt.year)['value'].max()
 
    return annual_max


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
    plt.ylabel('Annual maximum rainfall (mm)')
    plt.title('Return level plot — Trimpley rainfall')
    plt.legend()
    plt.tight_layout()
    plt.show()


### GEV FIT
def fit_gev(annual_max: pd.Series) -> tuple[float, float, float]:
    return genextreme.fit(annual_max.values)

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
    annual_max = load_annual_maxima('data/Trimpley-rainfall-daily-Qualified.csv')
    params = fit_gev(annual_max) # c (shape), loc (location), scale 

    print(f"Years with a valid annual maximum: {len(annual_max)}")
    print(f"Year range: {annual_max.index.min()}-{annual_max.index.max()}")
    print(f"Largest annual maximum: {annual_max.max():.1f} mm "
          f"(year {annual_max.idxmax()})")
    print(f"Smallest annual maximum: {annual_max.min():.1f} mm "
          f"(year {annual_max.idxmin()})")
    expected_years = set(range(annual_max.index.min(), annual_max.index.max() + 1))
    missing_years = sorted(expected_years - set(annual_max.index))
    print(f"Years with no valid annual maximum: {missing_years}")

    statistic, p_value = gof_test_gev(annual_max, params)
    print(f"Goodness of fit test: statistic={statistic:.3f}, p-value={p_value:.3f}")
    plot_return_levels(annual_max, params)
