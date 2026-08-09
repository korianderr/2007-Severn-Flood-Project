# Severn Flood Project — Log

## Project goal (clarified)
ML emulator (CNN/U-Net) for flood inundation, trained on synthetic data from a
minimal physical flood simulator + EA hazard maps. Feeds into a Monte Carlo
catastrophe risk model over extreme-value rainfall scenarios, producing an
exceedance-probability curve and expected annual damage estimate.
Case study: July 2007 Severn floods, Tewkesbury-Gloucester reach.

**Scope note:** the flood simulator is deliberately minimal (2D diffusive-wave /
cellular-automaton, not a full hydrodynamic solver). Its only job is generating
varied, plausible-looking training scenarios for the emulator — it does not
need to be precisely calibrated.

## Week 1

### Terrain data (DEM)
- Reach of interest: Tewkesbury to Gloucester, OS grid squares SO81, SO82, SO83.
- Downloaded 1m LIDAR Composite DTM tiles from EA Survey Data Download portal
  (environment.data.gov.uk/survey), OGL licence, no login required.
- Merged and downsampled to 5m resolution in QGIS (OSGeo4W install):
  - Raster > Miscellaneous > Merge to stitch tiles.
  - Raster > Projections > Warp (Reproject) to resample 1m -> 5m.
  - Compression: DEFLATE + PREDICTOR=3 (correct predictor for float elevation
    data — PREDICTOR=2 is for integers and barely compresses floats).
- Result: `merged_5m.tif`, EPSG:27700, bounds 380000-390000 E / 210000-240000 N.
- GeoTIFF isn't a supported upload type in this chat interface — used GitHub
  (via GitHub Desktop, since the file exceeded the 25MB browser upload limit)
  as a relay to get it into the working environment.
- QC checked: no gaps/seams in the mosaic, elevation range (1.4m-283m)
  sensible, isolated bumps confirmed as real hills (Robinswood Hill, Churchdown
  Hill) rather than artifacts.
- Clipped to a tighter reach corridor: `reach_clip.tif`.

### Historical event data
- Two EA APIs exist: `flood-monitoring` (near-real-time only) vs `hydrology`
  (long-term quality-checked archive) — used `hydrology` for 2007 data.
- Gauges used:
  - **Haw Bridge** (River Severn, near Gloucester) — level, 15-min. All-time
    record 6.228m recorded 08:00 23 July 2007.
  - **Trimpley** — rainfall, 15-min, upstream on the Severn near Bewdley.
- Gotcha: "all-measures" CSV exports bundle every measure type (15-min, daily,
  flow, level, etc.) into one file distinguished only by a `measure` column —
  plotting `value` directly without filtering by `measure` first produces a
  meaningless zigzag. Must filter to the specific measure URL before plotting.
- Result after filtering: rainfall peaks ~20-21 July, level responds with
  roughly a 1-day lag, peaking just above 6m around 23 July — matches the real
  recorded event closely. Good validation that the data pull is correct.

### Still to decide / do
- Upstream boundary condition for the solver: decided to drive it directly off
  a level/stage hydrograph at the Tewkesbury end (simpler than converting to
  flow), rather than sourcing a separate flow gauge.
- Write the minimal CA/diffusive-wave solver (kept deliberately small per
  scope note above).
- One sanity-check run (water flows downhill sensibly, pools where expected)
  — not a calibration exercise.

## Up next — Week 2
- Parameterize solver inputs (rainfall magnitude/shape, etc.) and generate a
  training dataset of synthetic flood scenarios.
- Build CNN/U-Net emulator trained on that data, validated against EA hazard
  maps / historic flood outlines.
