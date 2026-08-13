"""
generate_scenarios.py

Generates synthetic training examples across a range of boundary WSEs, since the CNN emulator needs far
more examples than the single real 2007 flood provides.
"""

import csv
from pathlib import Path
import numpy as np
import rasterio

from solver import bathtub_fill

def make_wse_range(low: float, high: float, n_scenarios: int) -> np.ndarray:
    """
    Takes as input a low and high wse to generate evenly spaced wse's within that range
    """
    return np.linspace(low, high, n_scenarios)

def generate_scenarios(z: np.ndarray, wse_range: np.ndarray, output_dir: str) -> str:
    """
    Takes as input the ground elevations, z, the desired wse range to generate scenarios for, wse_range,
    and the directory to save the scenarios to, output_dir. 

    Uses bathtub_fill to generate scenarios at each wse in the range provided, and save them to a file
    before writing their scenario id, wse, and filename to manifest.csv.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario_id", "boundary_wse", "filename"])

        for scenario_id, boundary_wse in enumerate(wse_range):
            h_final = bathtub_fill(z, boundary_wse=boundary_wse)
            file = f"scenario_{scenario_id}.npy"
            path = out_dir / file
            np.save(path, h_final)
            writer.writerow([scenario_id, boundary_wse, file])

    return str(manifest_path)

if __name__ == "__main__":
    with rasterio.open("data/reach_clip_25m.tif") as src:
        z = src.read(1)

    wse_range = make_wse_range(low=9.5, high=14, n_scenarios=25)
    manifest = generate_scenarios(z, wse_range, output_dir="data/scenarios")
    print("Manifest written to:", manifest)
