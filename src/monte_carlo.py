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
import matplotlib.pyplot as plt

from dataset import normalise, load_datasets
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
            physical_max = np.maximum(boundary_wse - dataset.z, 0)
            depth = np.clip(depth, 0, physical_max) # Can't exceed boundary_wse - z (see solver.py)
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


def flooded_area_per_year(maxima, wse_grid, depth_stack, mask, thresh=0.05, cell_area=625.0):
    """
    Reduce each simulated year to one hazard metric, the flooded area.

    Each year's peak level is mapped to the nearest cached depth grid,
    and that grid's flooded area is the year's value. The area is
    computed once per grid point rather than once per year so that
    the sum doesn't have to be repeated thousands of times.

    Years with no exceedance are recorded as zero.

    Args:
        maxima: output of simulate_annual_maxima() — annual peak level
            in mAOD, NaN in years with no exceedance.
        wse_grid: the WSE grid passed to build_depth_lookup().
        depth_stack: its output, shape (len(wse_grid), H, W).
        thresh: wet/dry depth threshold in metres. Matches the value
            used in evaluate.py so hazard extent is defined
            consistently across the project.
        cell_area: area of one DEM cell in m². 625 = 25m x 25m, the
            resolution of reach_clip_25m.tif.

    Returns:
        (areas, area_by_grid):
            areas: np.ndarray, shape (len(maxima),) — flooded area in
                m² for each simulated year, 0 in non-exceedance years.
            area_by_grid: np.ndarray, shape (len(wse_grid),) — flooded
                area at each grid point, returned so monotonicity in
                WSE can be checked without recomputing.
    """
    area_by_grid = np.array([((d > thresh) & mask).sum() * cell_area for d in depth_stack])

    # Drop years of no exceedance
    out = np.zeros(len(maxima))
    valid = ~np.isnan(maxima)

    # Map each valid year to nearest grid point
    idx = np.abs(wse_grid[None, :] - maxima[valid][:, None]).argmin(axis=1)

    # Get correct area for each year
    out[valid] = area_by_grid[idx]

    return out, area_by_grid


def exceedance_curve(annual_values):
    """
    Convert annual values into an exceedance-probability curve.

    Sorted descending. Annual exceedance probability of an
    exceedance at rank k has exceedance probability of k/n
    where n is the number of simualated years. 
    
    Args:
        annual_values: one value per simulated year (m², or currency
            once exposure is added).

    Returns:
        (aep, values): aep ascending from 1/n to 1, values descending.
            Return period is 1/aep.
    """
    n = len(annual_values)
    values = np.sort(annual_values)[::-1]
    aep = np.arange(1, n + 1) / n
    return aep, values


if __name__ == "__main__":
    import time

    from model import FloodUNet
    from solver import bathtub_fill

    u, sigma, lam = PARAM_CASES['central']
    n_years = 100000
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

    # 
    areas, area_by_grid = flooded_area_per_year(maxima, wse_grid, depth_stack, train_dataset.mask)

    print(f"zero years: {(areas == 0).mean():.3f}  (expect ~0.18)")
    print(f"max: {areas.max()/1e6:.1f} km²   mean: {areas.mean()/1e6:.2f} km²")
    print(f"monotonic in level: {np.all(np.diff(area_by_grid) >= 0)}")

    print(f"area range: {area_by_grid[0]/1e6:.1f} to {area_by_grid[-1]/1e6:.1f} km²")
    print(f"mask covers: {train_dataset.mask.sum()*625/1e6:.1f} km² of 230 km²")

    true_areas = np.array([
        ((bathtub_fill(train_dataset.z, boundary_wse=w) > 0.05) & train_dataset.mask).sum() * 625.0
        for w in wse_grid
    ])
    for i in [0, 20, 40, 59]:
        print(f"wse {wse_grid[i]:.2f}: emulator {area_by_grid[i]/1e6:6.1f}  "
            f"solver {true_areas[i]/1e6:6.1f} km²")
    print(f"mean abs area error: {np.abs(area_by_grid-true_areas).mean()/1e6:.1f} km²")

    # Plot exceedance curve
    aep, values = exceedance_curve(areas)
    plt.figure(figsize=(7, 5))
    plt.semilogx(1 / aep, values / 1e6)
    plt.xlabel('Return period (years)')
    plt.ylabel('Flooded area (km²)')
    plt.title('Hazard exceedance curve — central case')
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.show()
