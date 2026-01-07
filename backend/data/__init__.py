"""
Flood Forecasting Data Module
=============================
Contains data generation, simulation, and preprocessing utilities
for the AI-powered flood prediction system.
"""

from .hydrological_simulator import (
    HydrologicalSimulator,
    INDIA_RIVERS,
    generate_all_river_datasets
)

__all__ = [
    "HydrologicalSimulator",
    "INDIA_RIVERS", 
    "generate_all_river_datasets"
]
