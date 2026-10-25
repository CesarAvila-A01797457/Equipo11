"""Limpia los viajes y prepara la demanda diaria por estación."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from loguru import logger
import mlflow
import mlflow.data
import pandas as pd
import typer

from MiBici.config import PROCESSED_DATA_DIR, PROJ_ROOT, RAW_DATA_DIR
from MiBici.data_cleaning_stages import DEFAULT_PIPELINE, coerce_and_validate_types, drop_duplicate_trip_ids, drop_short_same_station_trips, run_pipeline

app = typer.Typer(help="Limpia y prepara los datos de viajes de MiBici.")
CATALOG_NAME = "nomenclatura_2026_09.csv"
CATALOG_URL = "https://mibici.net/site/assets/files/1118/nomenclatura_2026_09.csv"
MANIFEST_NAME = "manifest.csv"
TRIP_COLUMNS = [
    "Viaje_Id",
    "Inicio_del_viaje",
    "Fin_del_viaje",
    "Origen_Id",
    "Destino_Id",
]
OUTPUT_NAME = "mibici_salidas_estacion_fecha.csv"
QUALITY_REPORT_NAME = "mibici_quality_report.json"


def csv_encoding(path: Path) -> str:
    """Detecta la codificación para poder leer cada CSV."""
    with path.open("rb") as file:
        sample = file.read(128_000)
    try:
        sample.decode("utf-8-sig")
        return "utf-8-sig"
    except UnicodeDecodeError:
        return "cp1252"


def load_dataframes(
    raw_dir: Path = RAW_DATA_DIR / "mibici",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Consolida los CSV mensuales y carga el catálogo en otro DataFrame."""
    trip_files = sorted(raw_dir.glob("datos_abiertos_*.csv"))
    catalog_path = raw_dir / CATALOG_NAME
    if not trip_files:
        raise typer.BadParameter(f"No se encontraron CSV mensuales en {raw_dir}")
    if not catalog_path.is_file():
        raise typer.BadParameter(
            f"No se encontró el catálogo de estaciones: {catalog_path}"
        )

    logger.info("Carga de los CSV mensuales originales")
    logger.info(
        "Se combinarán {} archivos ({} a {}) en un DataFrame",
        len(trip_files),
        trip_files[0].stem,
        trip_files[-1].stem,
    )

    trip_frames = []
    progress_interval = max(1, (len(trip_files) + 9) // 10)
    for index, path in enumerate(trip_files, start=1):
        trip_frames.append(pd.read_csv(path, encoding=csv_encoding(path)))
        if index % progress_interval == 0 or index == len(trip_files):
            logger.info("Progreso: {}/{} archivos mensuales cargados", index, len(trip_files))

    trips_df = pd.concat(trip_frames, ignore_index=True, sort=False)

    logger.info("Carga del catálogo de estaciones")
    stations_df = pd.read_csv(catalog_path, encoding=csv_encoding(catalog_path))

    logger.info(
        "Viajes: {} filas y {} columnas: {}",
        len(trips_df),
        len(trips_df.columns),
        ", ".join(map(str, trips_df.columns)),
    )
    logger.info(
        "Estaciones: {} filas y {} columnas: {}",
        len(stations_df),
        len(stations_df.columns),
        ", ".join(map(str, stations_df.columns)),
    )

    logger.info("Los DataFrames se mantienen separados; no se unen entre sí.")
    return trips_df, stations_df


def prepare_trip_data(
    trips_df: pd.DataFrame,
    trip_files: list[Path],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Limpia el DataFrame consolidado y cuenta salidas por fecha y estación."""

    DEFAULT_PIPELINE = [
        coerce_and_validate_types,
        drop_duplicate_trip_ids,
        drop_short_same_station_trips,
    ]
    cleaning = run_pipeline(trips_df, DEFAULT_PIPELINE)
    if cleaning.frame.empty:
        raise typer.BadParameter("No quedaron viajes válidos para preparar el dataset.")

    # La tabla final contiene una fila por fecha y estación de origen.
    daily_trips = pd.DataFrame(
        {
            "fecha": cleaning.frame["Inicio_del_viaje"].dt.strftime("%Y-%m-%d"),
            "estacion_id": cleaning.frame["Origen_Id"].astype("int64"),
        }
    )
    prepared_df = (
        daily_trips.groupby(["fecha", "estacion_id"], as_index=False)
        .size()
        .rename(columns={"size": "salidas"})
        .sort_values(["fecha", "estacion_id"])
        .reset_index(drop=True)
    )
    periods = [
        path.stem.removeprefix("datos_abiertos_").split("-")[0]
        for path in trip_files
    ]
    # El reporte deja constancia del volumen, las exclusiones y los archivos de origen.
    report = {
        "source": "MiBici Open Data",
        "source_page": "https://mibici.net/en/open-data/",
        "period_start": periods[0],
        "period_end": periods[-1],
        "input_files": len(trip_files),
        "files": [
            {"filename": path.name, "bytes": path.stat().st_size}
            for path in trip_files
        ],
        "rows_read": cleaning.rows_read,
        "rows_kept": cleaning.rows_kept,
        "rows_removed": cleaning.rows_removed,
        "rows_removed_total": cleaning.rows_removed_total,
        "timestamp_formats": cleaning.timestamp_formats,
        "output_rows": len(prepared_df),
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cleaning_rules": [
            "Se usan Viaje_Id, Inicio_del_viaje, Fin_del_viaje, Origen_Id y Destino_Id.",
            *cleaning.cleaning_rules,
            "Se conservan los picos de demanda; no hay un criterio validado para descartarlos.",
            "Los CSV originales no se modifican.",
        ],
    }
    return prepared_df, report


@app.command()
def run(
    raw_dir: Annotated[
        Path, typer.Option(help="Carpeta con los CSV originales de MiBici.")
    ] = RAW_DATA_DIR / "mibici",
    experiment_name: Annotated[
        str, typer.Option(help="Nombre del experimento en MLflow.")
    ] = "mibici-data-loading",
) -> None:
    """Prepara la demanda diaria, conserva el catálogo y registra ambos en MLflow."""
    trip_files = sorted(raw_dir.glob("datos_abiertos_*.csv"))
    catalog_path = raw_dir / CATALOG_NAME
    if not trip_files:
        raise typer.BadParameter(f"No se encontraron CSV mensuales en {raw_dir}")
    if not catalog_path.is_file():
        raise typer.BadParameter(
            f"No se encontró el catálogo de estaciones: {catalog_path}"
        )

    # 1. Consolidar los CSV y cargar el catálogo de estaciones.
    logger.info("ETAPA 1/3: Carga y consolidación de los datos")
    raw_trips_df, stations_df = load_dataframes(raw_dir)

    # 2. Limpiar los viajes y crear la tabla diaria para el análisis.
    logger.info("ETAPA 2/3: Limpieza y preparación de los viajes")
    trips_df = raw_trips_df[TRIP_COLUMNS].copy()
    del raw_trips_df
    prepared_df, quality_report = prepare_trip_data(trips_df, trip_files)
    del trips_df

    output_path = PROCESSED_DATA_DIR / OUTPUT_NAME
    report_path = PROCESSED_DATA_DIR / QUALITY_REPORT_NAME
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    prepared_df.to_csv(output_path, index=False)
    report_path.write_text(
        json.dumps(quality_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info(
        "Viajes preparados: {} filas (fecha, estacion_id, salidas); "
        "se leyeron {} viajes y se excluyeron {}",
        len(prepared_df),
        quality_report["rows_read"],
        quality_report["rows_removed_total"],
    )
    for reason, count in quality_report["rows_removed"].items():
        logger.info("Registros excluidos por {}: {}", reason, count)
    logger.info(
        "Catálogo de estaciones: {} filas y {} columnas",
        len(stations_df),
        len(stations_df.columns),
    )
    logger.info("Dataset preparado: {}", output_path)
    logger.info("Reporte de calidad: {}", report_path)

    tracking_db = (PROJ_ROOT / "data" / "mlflow.db").resolve()
    artifact_dir = (PROJ_ROOT / "data" / "mlartifacts").resolve()
    # 3. Registrar en MLflow los datos preparados y los archivos que explican su origen.
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db}")
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name, artifact_location=artifact_dir.as_uri()
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)

    logger.info("ETAPA 3/3: Registro del dataset preparado y el catálogo en MLflow")
    with mlflow.start_run(run_name="prepare-mibici-demand") as run:
        demand_dataset = mlflow.data.from_pandas(
            prepared_df,
            source=str(output_path.resolve()),
            name="mibici_salidas_estacion_fecha",
        )
        stations_dataset = mlflow.data.from_pandas(
            stations_df,
            source=str(catalog_path.resolve()),
            name="mibici_estaciones_2026_09",
        )
        mlflow.log_input(demand_dataset, context="prepared")
        mlflow.log_input(stations_dataset, context="source")
        mlflow.log_param("trip_source", "CSV mensuales MiBici")
        mlflow.log_param("trip_source_file_count", len(trip_files))
        mlflow.log_param(
            "trip_file_period", f"{trip_files[0].stem} to {trip_files[-1].stem}"
        )
        mlflow.log_param(
            "preparation",
            "validación, viajes cortos en la misma estación y conteo diario",
        )
        mlflow.log_param("station_catalog", CATALOG_NAME)
        mlflow.log_param("station_catalog_url", CATALOG_URL)
        mlflow.log_metric("trip_rows_read", quality_report["rows_read"])
        mlflow.log_metric("trip_rows_kept", quality_report["rows_kept"])
        mlflow.log_metric("trip_rows_removed", quality_report["rows_removed_total"])
        mlflow.log_metric("daily_station_rows", len(prepared_df))
        mlflow.log_metric("station_rows", len(stations_df))
        for reason, count in quality_report["rows_removed"].items():
            mlflow.log_metric(f"removed_{reason}", count)

        mlflow.log_artifact(str(output_path), artifact_path="prepared")
        mlflow.log_artifact(str(report_path), artifact_path="prepared")
        mlflow.log_artifact(str(Path(__file__).resolve()), artifact_path="source")

        manifest_path = raw_dir / MANIFEST_NAME
        if manifest_path.is_file():
            mlflow.log_artifact(str(manifest_path), artifact_path="source")
        else:
            logger.warning("No se encontró el manifiesto de descargas: {}", manifest_path)
        mlflow.log_artifact(
            str(PROJ_ROOT / "references" / "mibici_urls_2014_2026.txt"),
            artifact_path="source",
        )

    logger.success("Datos preparados y registrados en MLflow.")
    logger.info("Experimento MLflow: {}", experiment_name)
    logger.info("ID de ejecución: {}", run.info.run_id)
    logger.info("Base de MLflow: {}", tracking_db)
    logger.info("Artefactos de MLflow: {}", artifact_dir)
    logger.info("Catálogo utilizado: {}", CATALOG_NAME)
    logger.info("Dataset preparado y catálogo disponibles como entradas del experimento.")


if __name__ == "__main__":
    app()
