"""Quita viajes muy cortos que regresan a la misma estación."""

import pandas as pd

from MiBici.data_cleaning_stages.coerce_types import parse_start_time
from MiBici.data_cleaning_stages.pipeline import StageOutcome

SHORT_TRIP_LIMIT = pd.Timedelta(minutes=2)


def drop_short_same_station_trips(
    trips_df: pd.DataFrame,
) -> tuple[pd.DataFrame, StageOutcome]:
    """Elimina viajes de menos de 2 minutos con el mismo origen y destino.

    ``Inicio_del_viaje`` ya debe ser una fecha válida. ``Fin_del_viaje`` se
    interpreta con los mismos formatos. Si la llegada no se puede leer, o
    ocurre antes de la salida, el viaje se conserva. Las demás columnas
    se conservan.
    """
    end_time, _formats = parse_start_time(trips_df["Fin_del_viaje"])
    destination_id = pd.to_numeric(trips_df["Destino_Id"], errors="coerce")
    same_station = (
        destination_id.notna()
        & destination_id.gt(0)
        & destination_id.mod(1).eq(0)
        & destination_id.eq(trips_df["Origen_Id"])
    )
    duration = end_time - trips_df["Inicio_del_viaje"]
    short_trip = end_time.notna() & duration.ge(pd.Timedelta(0)) & duration.lt(SHORT_TRIP_LIMIT)
    remove = (same_station & short_trip).fillna(False)

    cleaned = trips_df.loc[~remove].copy()
    return cleaned, StageOutcome(
        rows_removed={"short_same_station_trips": int(remove.sum())},
        rules=(
            "Se eliminan viajes de menos de 2 minutos que salen y llegan a la misma estación.",
        ),
    )
