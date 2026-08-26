"""
exposure.py

Builds exposure layer for risk model: OSM building footprints
rasterised onto the 25m DEM grid, so each cell carries a value at risk.
"""
import rasterio
from rasterio.warp import transform_bounds

def dem_bounds_wgs_84(dem_path="data/reach_clip_25m.tif"):
    """
    Return the DEM's bounding box in ESPG:4326 (lat/lon).

    OSM data is served in WGS84 while this project works in ESPG:27700
    (British National Grid).

    Returns:
        West, south, east, north, in degrees.
    """
    with rasterio.open(dem_path) as src:
        return transform_bounds("EPSG:27700", "EPSG:4326", *src.bounds)

if __name__ == "__main__":
    west, south, east, north = dem_bounds_wgs_84()
    print(f"lat/lon: N={north:.4f} S={south:.4f} E={east:.4f} W={west:.4f}")
