"""
damage.py

Vulnerability layer for the risk model: converts predicted flood depth
into monetary damage using a depth-damage curve and the rasterised
exposure grid from exposure.py.
"""

import numpy as np


def damage_per_grid(depth_stack, exposure, max_depth=3.0):
    """
    Total damage at each WSE grid point.

    Damage ratio is min(depth / max_depth, 1) per cell, linear up to
    max_depth, total loss beyond. Computed once per grid point rather
    than once per simulated year.

    Args:
        depth_stack: output of build_depth_lookup(), shape
            (n_levels, H, W), depth in metres.
        exposure: (H, W) value at risk in GBP, from exposure.py.
        max_depth: depth in metres at which a property is considered a
            total loss.

    Returns:
        np.ndarray, shape (n_levels,) — total damage in GBP at each
        grid point.
    """

    ratios = np.clip(depth_stack / max_depth, 0, 1)
    return (ratios * exposure).sum(axis=(1, 2))


if __name__ == "__main__":
    import torch
    from dataset import load_datasets
    from model import FloodUNet
    from monte_carlo import build_depth_lookup

    exposure = np.load("data/exposure.npy")
    train_dataset, _, _ = load_datasets()

    model = FloodUNet()
    model.load_state_dict(torch.load("data/flood_unet.pt"))
    model.eval()

    wse_grid = np.linspace(10.4, 15.0, 90)
    depth_stack = build_depth_lookup(model, train_dataset, wse_grid)

    damages = damage_per_grid(depth_stack, exposure)
    for i in np.linspace(0, 89, 5).astype(int):
        print(f"wse {wse_grid[i]:5.2f}: £{damages[i]/1e9:.2f}bn")
    print(f"monotonic: {np.all(np.diff(damages) >= 0)}")
