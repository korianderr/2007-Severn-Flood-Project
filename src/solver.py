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
) -> np.ndarray:
    """
    Move water between neighbouring cells for one timestep.

    Water moves from cells with a higher water-surface elevation
    to cells with a lower water-surface elevation.

    The total amount leaving a cell is limited so that a cell
    can never send away more water than it contains.
    """

    # Water-surface elevation = ground elevation + water depth
    wse = z + h

    # Calculate the desired flow between each pair of neighbours.
    #
    # Each flow array represents water leaving the first cell
    # and moving into the second cell.

    # Flow upwards
    diff_up = wse[1:, :] - wse[:-1, :]
    flow_up = np.maximum(k * diff_up, 0)

    # Flow downwards
    diff_down = wse[:-1, :] - wse[1:, :]
    flow_down = np.maximum(k * diff_down, 0)

    # Flow left
    diff_left = wse[:, 1:] - wse[:, :-1]
    flow_left = np.maximum(k * diff_left, 0)

    # Flow right
    diff_right = wse[:, :-1] - wse[:, 1:]
    flow_right = np.maximum(k * diff_right, 0)

    # Calculate how much each cell wants to send out.
    total_out = np.zeros_like(h)

    total_out[1:, :] += flow_up
    total_out[:-1, :] += flow_down
    total_out[:, 1:] += flow_left
    total_out[:, :-1] += flow_right

    # A cell cannot send away more water than it contains.
    scale = np.ones_like(h)

    cells_with_outflow = total_out > 0

    scale[cells_with_outflow] = np.minimum(
        1.0,
        h[cells_with_outflow] / total_out[cells_with_outflow]
    )

    # Apply the scaling to every outgoing flow.
    flow_up *= scale[1:, :]
    flow_down *= scale[:-1, :]
    flow_left *= scale[:, 1:]
    flow_right *= scale[:, :-1]

    # Now calculate the net change in water depth.
    delta = np.zeros_like(h)

    # Up
    delta[1:, :] -= flow_up
    delta[:-1, :] += flow_up

    # Down
    delta[:-1, :] -= flow_down
    delta[1:, :] += flow_down

    # Left
    delta[:, 1:] -= flow_left
    delta[:, :-1] += flow_left

    # Right
    delta[:, :-1] -= flow_right
    delta[:, 1:] += flow_right

    # Update the water depth.
    h_new = h + delta

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

        # Downstream boundary:
        # water reaching the bottom edge leaves the domain.
        h[-1, :] = 0

        if step % 10 == 0:
            max_idx = np.unravel_index(np.argmax(h), h.shape)

            print(
                f"step {step:3d} | "
                f"max depth = {h.max():.3f} m | "
                f"ground = {z[max_idx]:.3f} m | "
                f"WSE = {z[max_idx] + h[max_idx]:.3f} m"
            )

    return h