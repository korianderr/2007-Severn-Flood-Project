"""
main.py

Entry point — orchestrates the data ingestion and visualization steps.
Solver and emulator calls will be added here as those modules come online.
"""

from src.data_ingestion import load_measure
from src.visualise import plot_rainfall_and_level

level_df = load_measure('data/Haw-Bridge-level-15min-Qualified.csv')
rain_df = load_measure('data/Trimpley-rainfall-15min-Qualified.csv')

plot_rainfall_and_level(rain_df, level_df)
