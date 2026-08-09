"""
data_ingestion.py

Loads and cleans raw data pulled from the EA hydrology API
(environment.data.gov.uk/hydrology). Handles the "all-measures" CSV export
format, which bundles multiple measure types (15-min, daily, flow, level,
etc.) into a single file distinguished only by a `measure` column — see
project_log.md for why filtering on this is necessary before use.
"""

import pandas as pd


def load_measure(csv_path: str) -> pd.DataFrame:
    """
    Load a pre-filtered EA hydrology CSV (single measure type — e.g. the
    15-min Qualified level/rainfall exports used in this project).
    """
    df = pd.read_csv(csv_path)
    df['dateTime'] = pd.to_datetime(df['dateTime'])
    return df.sort_values('dateTime')


def stage_to_aod(stage_value: float, datum: float = 6.0) -> float:
    """
    Convert a local stage-datum reading (mASD) to metres above Ordnance
    Datum (mAOD), using the station's stageScale.datum offset.
    Haw Bridge (station 2057): datum = 6 (from station metadata API).
    """
    return stage_value + datum
