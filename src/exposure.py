"""
exposure.py

Builds exposure layer for risk model: OSM building footprints
rasterised onto the 25m DEM grid, so each cell carries a value at risk.
"""
import rasterio
from rasterio.warp import transform_bounds
import osmnx as ox
import numpy as np
import matplotlib.pyplot as plt


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


def fetch_buildings(dem_path="data/reach_clip_25m.tif"):
    """
    Download OSM building footprints covering the DEM extent and
    reproject them to EPSG:27700.

    Hits the Overpass API live, so call it once and save the result.
    Overpass is a free service and rate-limits repeated queries.

    No tag filtering beyond building=True: this counts sheds, garages
    and industrial units alongside dwellings. Appropriate for a total
    exposed-value estimate, but it means a farm outbuilding carries the
    same notional value as a house.

    Returns:
        GeoDataFrame of building geometries in EPSG:27700.
    """

    west, south, east, north = dem_bounds_wgs_84(dem_path)
    buildings = ox.features_from_bbox(bbox=(west, south, east, north),
                                      tags={"building": True})
    return buildings.to_crs("EPSG:27700")


def rasterise_buildings(buildings, dem_path="data/reach_clip_25m.tif"):
    """
    Count building centroids per DEM cell.

    Centroids rather than polygon overlap: at 25m a cell is 625 m²,
    larger than most individual buildings, so the area-weighted version
    would add little. Counting (rather than a binary has-building mask)
    keeps settlement density, so a terrace contributes more than an
    isolated farmhouse.

    Returns:
        (H, W) int array of building counts, aligned to the DEM grid.
    """
    with rasterio.open(dem_path) as src:
        transform, height, width = src.transform, src.height, src.width

    pts = buildings.geometry.centroid
    rows, cols = rasterio.transform.rowcol(transform, pts.x.values, pts.y.values)
    rows, cols = np.asarray(rows), np.asarray(cols)

    inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    counts = np.zeros((height, width), dtype=np.int32)
    np.add.at(counts, (rows[inside], cols[inside]), 1)

    return counts

def build_exposure(counts, value_per_building=250_000.0):
    """
    Convert building counts to value at risk per cell.

    A value per building, applied to every OSM feature tagged building=True
    so sheds, garages and industrial units carry the same value as dwellings.
    Deliberately crude, and quoting it per-building keeps the assumption visible.

    Returns:
        (H, W) float array of value at risk in GBP.
    """
    return counts.astype(np.float64) * value_per_building


if __name__ == "__main__":
    # Check latitude and longitude
    west, south, east, north = dem_bounds_wgs_84()
    print(f"lat/lon: N={north:.4f} S={south:.4f} E={east:.4f} W={west:.4f}")

    # Check no. buildings
    buildings = fetch_buildings()
    print(f"features: {len(buildings)}")
    print(f"CRS: {buildings.crs}")
    print(f"bounds: {buildings.total_bounds}")

    # Check how many buildings placed, how many cells with buildings, max
    # per cell, mean elevation with and without buildings
    with rasterio.open("data/reach_clip_25m.tif") as src:
        z = src.read(1)

    counts = rasterise_buildings(buildings)
    exposure = build_exposure(counts)

    print(f"buildings placed: {counts.sum()} of {len(buildings)}")
    print(f"cells with buildings: {(counts > 0).sum()}")
    print(f"max per cell: {counts.max()}")
    print(f"mean elevation where buildings are: {z[counts > 0].mean():.1f} m")
    print(f"mean elevation overall: {z.mean():.1f} m")

    # Save counts and exposure
    np.save("data/building_counts.npy", counts)
    np.save("data/exposure.npy", exposure)

    # The mask is z <= 19m, matching dataset.py. The floodable subset
    # of the domain. Buildings outside it can never be hit by any
    # scenario, so the EAD rests entirely on the ones inside.
    mask = z <= 19.0

    print(f"buildings inside flood mask: {counts[mask].sum()} "
          f"of {counts.sum()} ({100*counts[mask].sum()/counts.sum():.1f}%)")
    print(f"occupied cells inside mask: {(counts > 0)[mask].sum()}")
    print(f"exposure inside mask: £{exposure[mask].sum()/1e9:.2f}bn "
          f"of £{exposure.sum()/1e9:.2f}bn")

    fig, axes = plt.subplots(1, 2, figsize=(9, 7))
    axes[0].imshow(z, cmap='terrain')
    axes[0].set_title('Elevation')
    axes[1].imshow(counts > 0, cmap='Reds')
    axes[1].set_title('Building cells')
    plt.tight_layout()
    plt.show()
