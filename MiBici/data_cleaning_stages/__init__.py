"""Etapas de limpieza que se encadenan en un pipeline."""

from MiBici.data_cleaning_stages.coerce_types import coerce_and_validate_types
from MiBici.data_cleaning_stages.drop_duplicate_trip_ids import drop_duplicate_trip_ids
from MiBici.data_cleaning_stages.drop_short_same_station_trips import (
    drop_short_same_station_trips,
)
from MiBici.data_cleaning_stages.pipeline import CleaningResult, StageOutcome, run_pipeline



__all__ = [
    "CleaningResult",
    "StageOutcome",
    "coerce_and_validate_types",
    "drop_duplicate_trip_ids",
    "drop_short_same_station_trips",
    "run_pipeline",
]
