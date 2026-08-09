"""
data_ingestion.py

Loads pre-filtered EA hydrology CSVs (single measure type — e.g. the
15-min Qualified level/rainfall exports used in this project). Note: the
raw "all-measures" export bundles multiple measure types into one file
distinguished by a `measure` column — if you ever switch back to that
export format, you'll need to filter on that column before use (see
project_log.md for why this matters).
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
