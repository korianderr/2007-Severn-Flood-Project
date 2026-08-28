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


def check_sampler(n_years=10000, seed=42):
    """Verify the sampler reproduces the fitted return levels."""
    u, sigma, lam = PARAM_CASES['central']
    maxima = simulate_annual_maxima(u, sigma, lam, n_years, seed=seed)
    print(f"NaN fraction: {np.isnan(maxima).mean():.3f}  (expect ~0.179)")
    valid = np.sort(maxima[~np.isnan(maxima)])[::-1]
    # Ranks scale with n_years: rank k has return period n_years/k.
    for T, expected in [(1000, 12.67), (100, 12.10), (10, 11.53)]:
        k = n_years // T
        print(f"T={T:>5} yr: {valid[k-1]:.2f} mAOD  (expect ~{expected})")


def validate_emulator_area(wse_grid, depth_stack, area_by_grid, z, mask):
    """
    Compare emulator flooded area against bathtub_fill across the grid.

    CSI measures pixel agreement. This measures total area, which is
    what the risk model actually consumes. The two can diverge.
    """
    true_areas = np.array([
        ((bathtub_fill(z, boundary_wse=w) > 0.05) & mask).sum() * 625.0
        for w in wse_grid
    ])
    for i in np.linspace(0, len(wse_grid) - 1, 4).astype(int):
        print(f"wse {wse_grid[i]:.2f}: emulator {area_by_grid[i]/1e6:6.1f}  "
              f"solver {true_areas[i]/1e6:6.1f} km²")
    print(f"mean abs area error: "
          f"{np.abs(area_by_grid - true_areas).mean()/1e6:.1f} km²")


def plot_ep_curves(wse_grid, depth_stack, mask, n_years=10000, seed=42):
    """
    EP curve under each POT parameter case, overlaid.

    Same seed across cases so the spread between curves reflects the
    parameters, not the random draw. That spread is the threshold
    uncertainty carried from pot_analysis.py — note it does NOT include
    shape uncertainty, since all three cases fix xi=0.
    """
    plt.figure(figsize=(7, 5))
    for name, (u, sigma, lam) in PARAM_CASES.items():
        maxima = simulate_annual_maxima(u, sigma, lam, n_years, seed=seed)
        areas, _ = flooded_area_per_year(maxima, wse_grid, depth_stack, mask)
        aep, values = exceedance_curve(areas)
        plt.semilogx(1 / aep, values / 1e6, label=name)

        above = (maxima[~np.isnan(maxima)] > 14.0).sum()
        print(f"{name:>8}: max {np.nanmax(maxima):.2f} mAOD, "
              f"{above} yr above emulator training range")

    plt.xlabel('Return period (years)')
    plt.ylabel('Flooded area (km²)')
    plt.title('Hazard exceedance curve — POT parameter sensitivity')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/ep_curve_sensitivity.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    import time
    from model import FloodUNet
    from solver import bathtub_fill

    check_sampler()

    # Emulator lookup. Grid spans all three parameter cases.
    train_dataset, _, _ = load_datasets()
    model = FloodUNet()
    model.load_state_dict(torch.load("data/flood_unet.pt"))
    model.eval()

    wse_grid = np.linspace(10.4, 15.0, 90)
    start = time.perf_counter()
    depth_stack = build_depth_lookup(model, train_dataset, wse_grid)
    print(f"{len(wse_grid)} emulator calls in {time.perf_counter()-start:.2f}s")

    _, area_by_grid = flooded_area_per_year(
        simulate_annual_maxima(*PARAM_CASES['central'], 10000, seed=42),
        wse_grid, depth_stack, train_dataset.mask
    )
    print(f"area range: {area_by_grid[0]/1e6:.1f} to {area_by_grid[-1]/1e6:.1f} km²")
    print(f"mask covers: {train_dataset.mask.sum()*625/1e6:.1f} km² of 230 km²")

    validate_emulator_area(wse_grid, depth_stack, area_by_grid,
                           train_dataset.z, train_dataset.mask)
    plot_ep_curves(wse_grid, depth_stack, train_dataset.mask)
