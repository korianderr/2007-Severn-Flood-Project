"""
solver.py

Minimal flood extent model using connectivity-constrained flood-fill
("bathtub fill"): a cell floods only if its elevation is below the target
water-surface elevation AND it's reachable from the upstream boundary via a
continuous path of other below-water-surface cells. This avoids the failure
mode of an earlier diffusive-wave version, which let isolated topographic
pits fill up independently of the actual drainage network (see
project_log.md). Deliberately simple by design — this generates training
scenarios for the ML emulator (Week 2), not a calibrated hydrodynamic model.
"""

import numpy as np
import scipy.ndimage

def bathtub_fill(z: np.ndarray, boundary_wse: float, boundary_row: int = 0) -> np.ndarray:
    """
    Compute flood depth via connectivity-constrained fill: a cell floods only
    if its ground elevation is below boundary_wse AND it's reachable from the
    boundary row through a continuous path of other cells also below
    boundary_wse. This prevents isolated low-lying pits (disconnected from
    the actual drainage network) from filling up independently — the failure
    mode found in the diffusive-wave approach (see project_log.md).

    Args:
        z: (rows, cols) DEM elevation array.
        boundary_wse: water-surface elevation (mAOD) to test against.
        boundary_row: which row represents the upstream boundary (default: 0,
            the top row / Tewkesbury end).

    Returns:
        (rows, cols) array of water depth — 0 everywhere not connected.
    """
    # Every cell low enough to be underwater at this water level, in isolation
    below_wse = z < boundary_wse

    # Label connected groups of below_wse cells (4-connectivity: up/down/left/right)
    labeled, _ = scipy.ndimage.label(below_wse)

    # Which labels actually touch the boundary row?
    boundary_labels = set(labeled[boundary_row, :][below_wse[boundary_row, :]])
    boundary_labels.discard(0)  # 0 = not-underwater cells, not a real group

    # Keep only cells belonging to a group connected to the boundary
    connected = np.isin(labeled, list(boundary_labels))

    depth = np.where(connected, boundary_wse - z, 0)
    return depth
