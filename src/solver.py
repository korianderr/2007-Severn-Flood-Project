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


def diffusive_wave_step(
    h: np.ndarray,
    z: np.ndarray,
    k: float = 0.1,
    leak_frac: float = 0.02,
) -> np.ndarray:
    """
    Move water between neighbouring cells for one timestep.

    Water moves from higher water-surface elevation to lower
    water-surface elevation. A small fraction of water is also
    removed from each cell to represent unresolved drainage.
    """

    # Water-surface elevation
    wse = z + h

    # Flow between neighbouring cells

    # Up
    diff_up = wse[1:, :] - wse[:-1, :]
    flow_up = np.maximum(k * diff_up, 0)

    # Down
    diff_down = wse[:-1, :] - wse[1:, :]
    flow_down = np.maximum(k * diff_down, 0)

    # Left
    diff_left = wse[:, 1:] - wse[:, :-1]
    flow_left = np.maximum(k * diff_left, 0)

    # Right
    diff_right = wse[:, :-1] - wse[:, 1:]
    flow_right = np.maximum(k * diff_right, 0)

    # Calculate net change
    delta = np.zeros_like(h)

    delta[1:, :] -= flow_up
    delta[:-1, :] += flow_up

    delta[:-1, :] -= flow_down
    delta[1:, :] += flow_down

    delta[:, 1:] -= flow_left
    delta[:, :-1] += flow_left

    delta[:, :-1] -= flow_right
    delta[:, 1:] += flow_right

    # Update depth
    h_new = h + delta

    # Simple drainage throughout the domain
    h_new *= (1 - leak_frac)

    return np.maximum(h_new, 0)


def run_simulation(
    dem_path: str,
    boundary_wse: float,
    n_steps: int,
    k: float = 0.1,
) -> np.ndarray:
    """
    Run the flood simulation.

    The top row is kept at the specified upstream water-surface
    elevation. The bottom row acts as an open downstream boundary.
    """

    with rasterio.open(dem_path) as src:
        z = src.read(1)

    # Start with the whole domain dry.
    h = np.zeros_like(z)

    for step in range(n_steps):

        # Upstream boundary:
        # force the top row to the specified water-surface elevation.
        h[0, :] = np.maximum(boundary_wse - z[0, :], 0)

        # Move water between neighbouring cells.
        h = diffusive_wave_step(h, z, k=k)

        if step % 10 == 0:
            max_idx = np.unravel_index(np.argmax(h), h.shape)

            print(
                f"step {step:3d} | "
                f"max depth = {h.max():.3f} m | "
                f"ground = {z[max_idx]:.3f} m | "
                f"WSE = {z[max_idx] + h[max_idx]:.3f} m"
            )

    return h