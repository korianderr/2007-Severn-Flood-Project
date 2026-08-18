"""
monte_carlo.py

Monte Carlo sampling of synthetic flood years from the fitted POT model
to feed boundary WSEs into emulator and build an exceedance vs.
probability curve.

The model has two independent halves: flood peaks arrive as a poisson
process at rate lam, and each peak's excess above the threshold u is
exponential with scale sigma.

Fitted parameters:
    u = 10.821 mAOD: threshold
    sigma = 0.248m: exponential scale
    lam = 1.72 / yr: 87 events over 50.6 year record

Why these values:
The shape parameter is fixed at zero (exponential tail rather than a
full GPD) because it isn't identifiable from this record. Above ~10.8 the
shape estimates scatter around zero with no consistent sign, so zero is
what the data supports once into the tail.

The threshold was chosen at the highest threshold within the well-behaved
region (before it starts fluctuating around zero). It gives 1.72
events/year (lam)

Limitations:
The implied return period of the 2007 event ranges from 23 to 576 years
across candidate thresholds, so it is not identifiable from this record.
Two flanking cases are therefore carried alongside the central one and
the EP curve is run under all three:

    low   u=10.517, sigma=0.437, lam=2.234
    high  u=10.897, sigma=0.196, lam=1.562

The reason the fit is awkward is that extreme value analysis is being
done on stage rather than flow. Once a river spills onto the floodplain,
level stops responding to discharge, so exceedances bunch tightly just
above threshold while a breakout event like 2007 sits far beyond which
understates sigma and so overstates rarity. 
"""

import numpy as np


# (u, sigma, lam) — see module docstring. Threshold-dependent: always
# quote and vary together, never mix across cases.
PARAM_CASES = {
    'low':     (10.517, 0.437, 2.234),
    'central': (10.821, 0.248, 1.720),
    'high':    (10.897, 0.196, 1.562),
}


def simulate_annual_maxima(u, sigma, lam, n_years, seed=None):
    """
    Simulate annual maximum river levels from the fitted POT model.

    Gives a poisson count of independent flood peaks, then that many
    exponential excesses above u. The year's maximum is u + largest excess.
    The years with no peaks are recorded as NaN.

    Args:
        u: threshold (mAOD).
        sigma: exponential scale (m), mean excess above u.
        lam: mean independent flood peaks per year.
        n_years: number of years to simulate.
        seed: passed to default_rng for reproducibility.

    Returns:
        np.ndarray, shape (n_years,) — annual maximum level in mAOD,
        NaN in years with no threshold exceedance.
    """
    rng = np.random.default_rng(seed)
    counts = rng.poisson(lam, n_years)

    maxima = np.full(n_years, np.nan)

    for year, n in enumerate(counts):
        if n > 0:
            excesses = rng.exponential(sigma, n)
            maxima[year] = u + excesses.max()

    return maxima


if __name__ == "__main__":
    u, sigma, lam = PARAM_CASES['central']
    n_years = 10000
    maxima = simulate_annual_maxima(u, sigma, lam, n_years, seed=42)

    print(f"NaN fraction: {np.isnan(maxima).mean():.3f}  (expect ~0.179)")

    valid = np.sort(maxima[~np.isnan(maxima)])[::-1]
    for k, expected in [(10, 12.67), (100, 12.10), (1000, 11.53)]:
        print(f"T={n_years//k:>5} yr: {valid[k-1]:.2f} mAOD  (expect ~{expected})")
