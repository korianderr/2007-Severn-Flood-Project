"""
extreme_value.py
 
Extreme value analysis for the risk model: fitting a distribution to
historical rainfall extremes so we can Monte Carlo sample synthetic storm
scenarios beyond what's in the historical record. Starts with annual maxima
extraction (GEV fit); the GEV fitting itself and Monte Carlo sampling will
be added here as they're built.
"""
 
import pandas as pd
from scipy.stats import genextreme
 
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
 
 
if __name__ == "__main__":
    annual_max = load_annual_maxima('data/Trimpley-rainfall-daily-Qualified.csv')
 
    print(f"Years with a valid annual maximum: {len(annual_max)}")
    print(f"Year range: {annual_max.index.min()}-{annual_max.index.max()}")
    print(f"Largest annual maximum: {annual_max.max():.1f} mm "
          f"(year {annual_max.idxmax()})")
    print(f"Smallest annual maximum: {annual_max.min():.1f} mm "
          f"(year {annual_max.idxmin()})")
    expected_years = set(range(annual_max.index.min(), annual_max.index.max() + 1))
    missing_years = sorted(expected_years - set(annual_max.index))
    print(f"Years with no valid annual maximum: {missing_years}")

    params = genextreme.fit(annual_max.values)
    print(params) # Returns c (-xi), loc, scale
