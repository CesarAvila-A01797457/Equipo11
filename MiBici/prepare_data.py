"""Carga los viajes y el catálogo de estaciones en DataFrames separados."""

from pathlib import Path
from typing import Annotated

from loguru import logger
import mlflow
import pandas as pd
import typer

from MiBici.config import PROJ_ROOT, RAW_DATA_DIR

app = typer.Typer(help="Carga los datos originales de viajes y estaciones de MiBici.")
CATALOG_NAME = "nomenclatura_2026_09.csv"
CATALOG_URL = "https://mibici.net/site/assets/files/1118/nomenclatura_2026_09.csv"
MANIFEST_NAME = "manifest.csv"


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
    """Carga los viajes mensuales y la nomenclatura sin modificar sus columnas."""
    trip_files = sorted(raw_dir.glob("datos_abiertos_*.csv"))
    catalog_path = raw_dir / CATALOG_NAME
    if not trip_files:
        raise typer.BadParameter(f"No se encontraron CSV mensuales en {raw_dir}")
    if not catalog_path.is_file():
        raise typer.BadParameter(f"No se encontró el catálogo de estaciones: {catalog_path}")

    logger.info("ETAPA 1/3: Carga de los CSV mensuales de viajes")
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

    logger.info("ETAPA 2/3: Carga del catálogo de estaciones")
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
    if "dpcapacity" in stations_df.columns:
        logger.info("La capacidad dpcapacity se conserva en el DataFrame de estaciones.")
    return trips_df, stations_df


@app.command()
def run(
    raw_dir: Annotated[
        Path, typer.Option(help="Carpeta con los CSV originales de MiBici.")
    ] = RAW_DATA_DIR / "mibici",
    experiment_name: Annotated[
        str, typer.Option(help="Nombre del experimento en MLflow.")
    ] = "mibici-data-loading",
) -> None:
    """Carga ambos DataFrames y registra la fuente y el esquema en MLflow."""
    trips_df, stations_df = load_dataframes(raw_dir)
    tracking_db = (PROJ_ROOT / "data" / "mlflow.db").resolve()
    artifact_dir = (PROJ_ROOT / "data" / "mlartifacts").resolve()
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db}")
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name, artifact_location=artifact_dir.as_uri()
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    trip_files = sorted(raw_dir.glob("datos_abiertos_*.csv"))

    logger.info("ETAPA 3/3: Registro de la versión de las fuentes en MLflow")
    with mlflow.start_run(run_name="load-mibici-dataframes") as run:
        mlflow.log_param("trip_source", "CSV mensuales MiBici")
        mlflow.log_param("trip_source_file_count", len(trip_files))
        mlflow.log_param(
            "trip_file_period", f"{trip_files[0].stem} to {trip_files[-1].stem}"
        )
        mlflow.log_param("station_catalog", CATALOG_NAME)
        mlflow.log_param("station_catalog_url", CATALOG_URL)
        mlflow.log_param("trip_columns", ", ".join(map(str, trips_df.columns)))
        mlflow.log_param("station_columns", ", ".join(map(str, stations_df.columns)))
        mlflow.log_metric("trip_rows", len(trips_df))
        mlflow.log_metric("trip_column_count", len(trips_df.columns))
        mlflow.log_metric("station_rows", len(stations_df))
        mlflow.log_metric("station_column_count", len(stations_df.columns))

        manifest_path = raw_dir / MANIFEST_NAME
        if manifest_path.is_file():
            mlflow.log_artifact(str(manifest_path), artifact_path="source")
        else:
            logger.warning("No se encontró el manifiesto de descargas: {}", manifest_path)
        mlflow.log_artifact(
            str(PROJ_ROOT / "references" / "mibici_urls_2014_2026.txt"),
            artifact_path="source",
        )

    logger.success("Datos cargados.")
    logger.info("Experimento MLflow: {}", experiment_name)
    logger.info("ID de ejecución: {}", run.info.run_id)
    logger.info("Base de MLflow: {}", tracking_db)
    logger.info("Artefactos de MLflow: {}", artifact_dir)
    logger.info("Catálogo utilizado: {}", CATALOG_NAME)
    logger.info("Los DataFrames están disponibles en memoria durante esta ejecución.")


if __name__ == "__main__":
    app()
