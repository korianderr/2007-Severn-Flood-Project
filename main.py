"""
main.py

Entry point — orchestrates data ingestion, visualization, and the solver.
"""

from src.data_ingestion import load_measure, stage_to_aod
from src.visualise import plot_rainfall_and_level
from src.solver import run_simulation
from src.dem_processing import clip_to_reach
import matplotlib.pyplot as plt
from src.dem_processing import downsample_dem
import rasterio
import numpy as np

# --- Data validation (from earlier in the project) ---
level_df = load_measure('data/Haw-Bridge-level-15min-Qualified.csv')
rain_df = load_measure('data/Trimpley-rainfall-15min-Qualified.csv')
plot_rainfall_and_level(rain_df, level_df)

# --- Solver test run ---
# Using the July 2007 peak stage reading (6.228m mASD), converted to mAOD
# via the Haw Bridge station datum (6.0, from station metadata API).
# See project_log.md for the datum lookup process.
PEAK_STAGE_2007 = 6.228
BOUNDARY_WSE = stage_to_aod(PEAK_STAGE_2007)  # ~12.228 mAOD

N_STEPS = 10000 # Sanity check before scaling up to 500+ steps for the ML training data generation.

downsample_dem('data/reach_clip.tif', 'data/reach_clip_25m.tif', factor=5)
h_final = run_simulation('data/reach_clip_25m.tif', boundary_wse=BOUNDARY_WSE, n_steps=N_STEPS)

plt.figure(figsize=(10, 8))

plt.imshow(
    h_final,
    cmap='Blues',
    vmin=0,
    vmax=np.nanmax(h_final)
)

plt.colorbar(label='Water depth (m)')
plt.title('Simulated flood depth')
plt.xlabel('Column')
plt.ylabel('Row')

plt.show()

with rasterio.open("data/reach_clip_25m.tif") as src:
    z = src.read(1)

print("DEM min:", z.min())
print("DEM max:", z.max())
print("Top row min:", z[0, :].min())
print("Top row max:", z[0, :].max())
print("Initial boundary depth max:",
      np.maximum(BOUNDARY_WSE - z[0, :], 0).max())
