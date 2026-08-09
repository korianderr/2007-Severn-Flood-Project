"""
solver.py

Minimal 2D diffusive-wave / cellular-automaton flood spread model. Deliberately
simple by design (see project_log.md scope note) — this is not a calibrated
hydrodynamic solver, just a fast way to generate plausible flood-extent
scenarios as training data for the ML emulator (Week 2).

Physics: water moves from each grid cell to a lower neighbor, proportional to
the difference in water-surface elevation (ground height + water depth)
between them. No momentum, no time-dependent flow routing — just local
height differences pushing water downhill each timestep.
"""

import numpy as np
import rasterio


def diffusive_wave_step(h: np.ndarray, z: np.ndarray, k: float = 0.1, outflow_frac: float = 0.3) -> np.ndarray:
    """
    Advance water depth by one timestep using a simple diffusive-wave rule.

    Args:
        h: (rows, cols) array — current water depth at each cell (m).
        z: (rows, cols) array — ground elevation at each cell (m), same
            shape/grid as h (this is the DEM).
        k: fraction of the water-surface elevation difference that moves per
            step. Keep this small (roughly 0.05-0.15) — too large and the
            simulation becomes unstable (water oscillates instead of
            settling toward equilibrium).
        outflow_frac: fraction of the bottom row's water depth that exits
            the domain each step, representing the river continuing
            downstream past the study area (open boundary).

    Returns:
        (rows, cols) array — updated water depth after one step.
    """
    wse = z + h  # water surface elevation = ground height + water sat on top

    delta = np.zeros_like(h)  # net change in depth this step, built up below

    # Compare each cell to its neighbor ABOVE (row - 1)
    diff = wse[1:, :] - wse[:-1, :]           # self minus neighbor-above
    flow = np.where(diff > 0, k * diff, 0)    # only flows if self is higher
    delta[1:, :]  -= flow                     # water leaves the self cell
    delta[:-1, :] += flow                     # ...and arrives at the neighbor

    # Compare each cell to its neighbor BELOW (row + 1)
    diff = wse[:-1, :] - wse[1:, :]
    flow = np.where(diff > 0, k * diff, 0)
    delta[:-1, :] -= flow
    delta[1:, :]  += flow

    # Compare each cell to its neighbor LEFT (col - 1)
    diff = wse[:, 1:] - wse[:, :-1]
    flow = np.where(diff > 0, k * diff, 0)
    delta[:, 1:]  -= flow
    delta[:, :-1] += flow

    # Compare each cell to its neighbor RIGHT (col + 1)
    diff = wse[:, :-1] - wse[:, 1:]
    flow = np.where(diff > 0, k * diff, 0)
    delta[:, :-1] -= flow
    delta[:, 1:]  += flow

    h_new = h + delta

    # Open boundary: let water at the downstream edge drain out of the domain
    h_new[-1, :] *= (1 - outflow_frac)

    return np.maximum(h_new, 0)


def run_simulation(
    dem_path: str,
    boundary_wse: float,
    n_steps: int,
    k: float = 0.1,
) -> np.ndarray:
    """
    Run the diffusive-wave flood simulation for n_steps, forcing the top row
    of the grid (the Tewkesbury/upstream boundary) to boundary_wse at every
    step, and letting diffusive_wave_step() propagate water across the rest
    of the terrain.

    Args:
        dem_path: path to the DEM GeoTIFF to use as terrain (e.g. reach_clip.tif).
        boundary_wse: water surface elevation (m, same vertical datum as the
            DEM — mAOD) to force at the top row of the grid, every step.
        n_steps: number of timesteps to run.
        k: passed through to diffusive_wave_step — see its docstring.

    Returns:
        (rows, cols) array — final water depth after n_steps.
    """
    with rasterio.open(dem_path) as src:
        z = src.read(1)

    h = np.zeros_like(z)  # no flood at the start — everywhere dry

    for step in range(n_steps):
        # Force the top row (upstream boundary) to the target water surface
        # elevation each step: depth = target minus ground height, floored at 0.
        h[0, :] = np.maximum(boundary_wse - z[0, :], 0)

        h = diffusive_wave_step(h, z, k=k)

        if step % 10 == 0:
            print(f"step {step}, max depth: {h.max():.2f}")

    return h