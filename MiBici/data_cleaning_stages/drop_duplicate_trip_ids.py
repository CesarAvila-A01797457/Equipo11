"""Quita viajes repetidos dentro del mismo mes."""

import pandas as pd

from MiBici.data_cleaning_stages.pipeline import StageOutcome


def drop_duplicate_trip_ids(
    trips_df: pd.DataFrame,
) -> tuple[pd.DataFrame, StageOutcome]:
    """Elimina Viaje_Id repetidos dentro del mismo mes y conserva la primera fila.

    El mes sale de ``Inicio_del_viaje``, que ya debe ser una fecha válida.
    Las demás columnas se conservan.
    """
    trip_month = trips_df["Inicio_del_viaje"].dt.strftime("%Y-%m")
    duplicate = pd.DataFrame(
        {"month": trip_month, "trip_id": trips_df["Viaje_Id"]}
    ).duplicated(keep="first")
    cleaned = trips_df.loc[~duplicate].copy()
    return cleaned, StageOutcome(
        rows_removed={"duplicate_trip_ids": int(duplicate.sum())},
        rules=("Se eliminan identificadores de viaje duplicados dentro del mismo mes.",),
    )
