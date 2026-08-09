"""
dem_processing.py

Utilities for working with the merged DEM (digital elevation model) covering
the Tewkesbury-Gloucester reach. The initial tile merge and 1m->5m resample
were done manually in QGIS (see project_log.md for the exact steps/settings
used) — this module covers the parts that are cleaner to do in code, starting
with clipping the merged raster down to the actual reach of interest.
"""

import rasterio
from rasterio.windows import from_bounds


def clip_to_reach(
    input_path: str,
    output_path: str,
    bounds: tuple[float, float, float, float],
) -> None:
    """
    Clip a DEM GeoTIFF to a bounding box and save the result.

    Args:
        input_path: path to the source (merged) GeoTIFF.
        output_path: path to write the clipped GeoTIFF to.
        bounds: (minx, miny, maxx, maxy) in the source file's CRS
            (EPSG:27700 / British National Grid for this project — easting
            and northing in metres, not lat/lon).
    """
    minx, miny, maxx, maxy = bounds

    with rasterio.open(input_path) as src:
        window = from_bounds(minx, miny, maxx, maxy, src.transform)
        data = src.read(1, window=window)

        # Copy the source profile (CRS, dtype, etc.) and only update what's
        # actually changing: dimensions, transform, and compression settings.
        profile = src.profile.copy()
        profile.update({
            'height': data.shape[0],
            'width': data.shape[1],
            'transform': src.window_transform(window),
            # PREDICTOR=3 (not 2) — correct for floating-point elevation data,
            # gives much better compression than the integer predictor.
            'compress': 'deflate',
            'predictor': 3,
        })

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data, 1)