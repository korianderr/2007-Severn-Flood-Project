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
import torch

from dataset import normalise
from evaluate import denormalise


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


def build_depth_lookup(model, dataset, wse_grid):
    """
    Run the emulator once per WSE on the grid and cache the denormalised
    depth predictions. 
    
    Args:
        model: a FloodUNet in eval mode with weights loaded.
        dataset: any split from load_datasets() — supplies the DEM and
            the train-split normalisation constants.
        wse_grid: ascending array of boundary WSEs (mAOD) to evaluate.

    Returns:
        np.ndarray, shape (len(wse_grid), H, W) — predicted depth in
        metres, denormalised.
    """
    norm_z = normalise(dataset.z, dataset.z_min, dataset.z_max)

    depths = []
    with torch.no_grad():
        for boundary_wse in wse_grid:
            norm_wse = normalise(boundary_wse, dataset.wse_min, dataset.wse_max)
            wse_channel = np.full_like(norm_z, norm_wse, dtype=np.float32)

            input_array = np.stack([norm_z, wse_channel], axis=0)
            input_tensor = torch.from_numpy(input_array).float().unsqueeze(0)

            prediction = model(input_tensor)
            depth = denormalise(
                prediction, dataset.depth_min, dataset.depth_max
            ).squeeze().numpy()
            depth = np.maximum(depth, 0) # No negative
            depths.append(depth)

    return np.stack(depths)


def depth_for_level(level, wse_grid, depth_stack):
    """
    Look up the cached depth grid for the nearest WSE on the grid.

    Args:
        level: sampled annual maximum level (mAOD).
        wse_grid: the grid passed to build_depth_lookup().
        depth_stack: its output.

    Returns:
        (H, W) depth array in metres.
    """
    return depth_stack[np.abs(wse_grid - level).argmin()]


if __name__ == "__main__":
    import time
    from dataset import load_datasets
    from model import FloodUNet

    u, sigma, lam = PARAM_CASES['central']
    n_years = 10000
    maxima = simulate_annual_maxima(u, sigma, lam, n_years, seed=42)

    print(f"NaN fraction: {np.isnan(maxima).mean():.3f}  (expect ~0.179)")

    valid = np.sort(maxima[~np.isnan(maxima)])[::-1]
    for k, expected in [(10, 12.67), (100, 12.10), (1000, 11.53)]:
        print(f"T={n_years//k:>5} yr: {valid[k-1]:.2f} mAOD  (expect ~{expected})")

    ### EMULATOR
    train_dataset, _, _ = load_datasets()
    model = FloodUNet()
    model.load_state_dict(torch.load("data/flood_unet.pt"))
    model.eval()

    wse_grid = np.linspace(10.8, 13.6, 60)
    start = time.perf_counter()
    depth_stack = build_depth_lookup(model, train_dataset, wse_grid)
    print(f"{len(wse_grid)} emulator calls in {time.perf_counter()-start:.2f}s")

    totals = depth_stack.sum(axis=(1, 2))
    print(f"Monotonic in WSE: {np.all(np.diff(totals) > 0)}")
    print(f"Depth range: {depth_stack.min():.3f} to {depth_stack.max():.3f} m") 
