"""Convierte y valida los tipos de los campos clave de cada viaje."""

import pandas as pd

from MiBici.data_cleaning_stages.pipeline import StageOutcome


def parse_start_time(values: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    """Convierte los formatos de fecha usados en los CSV de MiBici."""
    text = values.astype("string").str.strip().replace("", pd.NA)
    iso_mask = text.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}", na=False)
    dmy_mask = text.str.match(r"^\d{1,2}[-/]\d{1,2}[-/]\d{4}", na=False)

    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    parsed.loc[iso_mask] = pd.to_datetime(
        text.loc[iso_mask], format="mixed", errors="coerce"
    )
    parsed.loc[dmy_mask] = pd.to_datetime(
        text.loc[dmy_mask], format="mixed", dayfirst=True, errors="coerce"
    )
    formats = {
        "year_month_day": int((iso_mask & parsed.notna()).sum()),
        "day_month_year": int((dmy_mask & parsed.notna()).sum()),
    }
    return parsed, formats


def coerce_and_validate_types(
    trips_df: pd.DataFrame,
) -> tuple[pd.DataFrame, StageOutcome]:
    """Convierte los campos clave y excluye los que quedan vacíos o inválidos.

    Cada fila se cuenta una sola vez, según el primer problema que se detecta:
    identificador de viaje, fecha de inicio y estación de origen. Las demás
    columnas se conservan.
    """
    trip_id = pd.to_numeric(trips_df["Viaje_Id"], errors="coerce")
    station_id = pd.to_numeric(trips_df["Origen_Id"], errors="coerce")
    start_time, timestamp_formats = parse_start_time(trips_df["Inicio_del_viaje"])

    valid_trip_id = trip_id.notna() & trip_id.gt(0) & trip_id.mod(1).eq(0)
    valid_station_id = station_id.notna() & station_id.gt(0) & station_id.mod(1).eq(0)
    valid_timestamp = start_time.notna()

    invalid_trip_id = ~valid_trip_id
    invalid_timestamp = valid_trip_id & ~valid_timestamp
    invalid_station_id = valid_trip_id & valid_timestamp & ~valid_station_id
    valid = valid_trip_id & valid_timestamp & valid_station_id

    cleaned = trips_df.loc[valid].copy()
    cleaned["Viaje_Id"] = trip_id.loc[valid].astype("int64")
    cleaned["Inicio_del_viaje"] = start_time.loc[valid]
    cleaned["Origen_Id"] = station_id.loc[valid].astype("int64")

    return cleaned, StageOutcome(
        rows_removed={
            "invalid_trip_id": int(invalid_trip_id.sum()),
            "invalid_timestamp": int(invalid_timestamp.sum()),
            "invalid_station_id": int(invalid_station_id.sum()),
        },
        rules=("Se excluyen campos clave vacíos o inválidos.",),
        details={"timestamp_formats": timestamp_formats},
    )
