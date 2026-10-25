"""Prepara el histórico MiBici por bloques, usando disco para deduplicar.

Conserva la regla del módulo original: primera fila válida por (mes, Viaje_Id).
No carga los viajes completos en memoria. Reutiliza el parser de fechas original.
"""

import csv
import hashlib
import json
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import mlflow
import mlflow.data
import pandas as pd
import typer
from loguru import logger
from MiBici.config import PROCESSED_DATA_DIR, PROJ_ROOT, RAW_DATA_DIR
from MiBici.prepare_data import (
    CATALOG_NAME,
    CATALOG_URL,
    MANIFEST_NAME,
    OUTPUT_NAME,
    QUALITY_REPORT_NAME,
    TRIP_COLUMNS,
    csv_encoding,
)

from MiBici.data_cleaning_stages.coerce_types import parse_start_time
from MiBici.data_cleaning_stages.drop_short_same_station_trips import (
    SHORT_TRIP_LIMIT,
    drop_short_same_station_trips,
)

app = typer.Typer(
    help="Prepara el histórico MiBici con lectura por bloques y SQLite temporal."
)
REMOVAL_KEYS = (
    "invalid_trip_id",
    "invalid_timestamp",
    "invalid_station_id",
    "duplicate_trip_ids",
    "short_same_station_trips",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def process_files(
    trip_files: list[Path], output_dir: Path, chunk_size: int = 50_000
) -> tuple[Path, dict]:
    """Limpia por bloques; deduplica incluso entre archivos y bloques del mismo mes."""
    if not trip_files:
        raise ValueError("No se encontraron CSV mensuales de viajes")
    if chunk_size <= 0:
        raise ValueError("chunk_size debe ser positivo")
    trip_files = sorted(map(Path, trip_files))
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_read = 0
    rows_kept = 0
    removed = dict.fromkeys(REMOVAL_KEYS, 0)
    formats = {"year_month_day": 0, "day_month_year": 0}
    files = []
    output_path = output_dir / OUTPUT_NAME

    # El directorio temporal es propio de esta ejecución, dentro de output_dir.
    # Si la ejecución falla, el CSV final anterior se conserva.
    with tempfile.TemporaryDirectory(
        prefix=".mibici-stream-", dir=output_dir
    ) as temporary:
        temporary_dir = Path(temporary).resolve()
        if not temporary_dir.is_relative_to(output_dir):
            raise RuntimeError("Directorio temporal fuera del destino esperado")
        connection = sqlite3.connect(temporary_dir / "dedup.sqlite")
        try:
            connection.execute(
                "PRAGMA cache_size=-8192"
            )  # Caché de páginas de aproximadamente 8 MiB.
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute(
                "CREATE TABLE trips (month TEXT NOT NULL, trip_id INTEGER NOT NULL, "
                "fecha TEXT NOT NULL, estacion_id INTEGER NOT NULL, keep_trip INTEGER NOT NULL, "
                "PRIMARY KEY(month, trip_id)) WITHOUT ROWID"
            )
            for index, path in enumerate(trip_files, 1):
                logger.info("[{}/{}] Procesando {}", index, len(trip_files), path.name)
                encoding = csv_encoding(path)
                file_rows = file_kept = 0
                with pd.read_csv(
                    path,
                    encoding=encoding,
                    usecols=TRIP_COLUMNS,
                    dtype="string",
                    chunksize=chunk_size,
                ) as reader:
                    for chunk in reader:
                        size = len(chunk)
                        rows_read += size
                        file_rows += size
                        trip_id = pd.to_numeric(chunk["Viaje_Id"], errors="coerce")
                        station_id = pd.to_numeric(chunk["Origen_Id"], errors="coerce")
                        start_time, chunk_formats = parse_start_time(
                            chunk["Inicio_del_viaje"]
                        )
                        for key, value in chunk_formats.items():
                            formats[key] += value
                        valid_id = (
                            trip_id.notna() & trip_id.gt(0) & trip_id.mod(1).eq(0)
                        ).fillna(False)
                        valid_station = (
                            station_id.notna()
                            & station_id.gt(0)
                            & station_id.mod(1).eq(0)
                        ).fillna(False)
                        valid_time = start_time.notna()
                        removed["invalid_trip_id"] += int((~valid_id).sum())
                        removed["invalid_timestamp"] += int(
                            (valid_id & ~valid_time).sum()
                        )
                        removed["invalid_station_id"] += int(
                            (valid_id & valid_time & ~valid_station).sum()
                        )
                        valid = valid_id & valid_time & valid_station
                        candidates = chunk.loc[valid].copy()
                        candidates["Viaje_Id"] = trip_id.loc[valid].astype("int64")
                        candidates["Origen_Id"] = station_id.loc[valid].astype("int64")
                        candidates["Inicio_del_viaje"] = start_time.loc[valid]
                        candidates["_month"] = start_time.loc[valid].dt.strftime("%Y-%m")
                        # Primero deduplicar; un viaje excluido también reserva su clave.
                        candidates = candidates.drop_duplicates(
                            ["_month", "Viaje_Id"], keep="first"
                        )
                        filtered, _ = drop_short_same_station_trips(candidates)
                        keep_mask = candidates.index.isin(filtered.index)
                        inserted_total = 0
                        with connection:
                            for keep_trip in (1, 0):
                                group = candidates.loc[keep_mask if keep_trip else ~keep_mask]
                                dates = group["Inicio_del_viaje"].dt.strftime("%Y-%m-%d")
                                before = connection.total_changes
                                connection.executemany(
                                    "INSERT OR IGNORE INTO trips VALUES (?, ?, ?, ?, ?)",
                                    (
                                        (date[:7], int(tid), date, int(station), keep_trip)
                                        for date, tid, station in zip(
                                            dates, group["Viaje_Id"], group["Origen_Id"]
                                        )
                                    ),
                                )
                                inserted = connection.total_changes - before
                                inserted_total += inserted
                                if keep_trip:
                                    rows_kept += inserted
                                    file_kept += inserted
                                else:
                                    removed["short_same_station_trips"] += inserted
                        removed["duplicate_trip_ids"] += int(valid.sum()) - inserted_total
                files.append(
                    {
                        "filename": path.name,
                        "bytes": path.stat().st_size,
                        "sha256": file_sha256(path),
                        "rows_read": file_rows,
                        "rows_kept": file_kept,
                        "encoding": encoding,
                    }
                )
                logger.info(
                    "{}: {} viajes leídos; {} retenidos",
                    path.name,
                    file_rows,
                    file_kept,
                )
            if rows_kept == 0:
                raise ValueError("No quedaron viajes válidos para preparar el dataset")

            logger.info(
                "Agregando conteos diarios desde disco; esta etapa puede tardar."
            )
            cursor = connection.execute(
                "SELECT fecha, estacion_id, COUNT(*) FROM trips "
                "WHERE keep_trip = 1 GROUP BY fecha, estacion_id ORDER BY fecha, estacion_id"
            )
            output_rows = total_departures = 0
            temporary_csv = temporary_dir / OUTPUT_NAME
            with temporary_csv.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["fecha", "estacion_id", "salidas"])
                while batch := cursor.fetchmany(5_000):
                    writer.writerows(batch)
                    output_rows += len(batch)
                    total_departures += sum(row[2] for row in batch)
            if total_departures != rows_kept or rows_read != rows_kept + sum(
                removed.values()
            ):
                raise RuntimeError("No coincide la conciliación de viajes y salidas")
            periods = [
                p.stem.removeprefix("datos_abiertos_").split("-")[0] for p in trip_files
            ]
            report = {
                "source": "MiBici Open Data",
                "source_page": "https://mibici.net/en/open-data/",
                "period_start": periods[0],
                "period_end": periods[-1],
                "input_files": len(trip_files),
                "files": files,
                "rows_read": rows_read,
                "rows_kept": rows_kept,
                "rows_removed": removed,
                "rows_removed_total": sum(removed.values()),
                "timestamp_formats": formats,
                "output_rows": output_rows,
                "created_at_utc": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "processing_mode": "chunked_csv_sqlite",
                "filter_version": "short_same_station_v1",
                "short_trip_limit_seconds": SHORT_TRIP_LIMIT.total_seconds(),
                "chunk_size": chunk_size,
                "conservation_check": total_departures == rows_kept,
                "cleaning_rules": [
                    "Se leen Viaje_Id, Inicio_del_viaje, Fin_del_viaje, Origen_Id y Destino_Id por bloques.",
                    "Se excluyen claves vacías, no positivas, no enteras o fechas inválidas.",
                    "Las causas de exclusión son excluyentes, en el orden del script original.",
                    "Se conserva el primer viaje válido por (mes de inicio, Viaje_Id).",
                    "La deduplicación abarca todos los bloques y archivos del mismo mes.",
                    "Después de deduplicar se excluyen viajes de duración >=0 y <2 minutos con origen igual a destino.",
                    "Fin inválido, duración negativa o destino inválido no activan el filtro.",
                    "Se conservan picos de demanda y no se infieren días con cero salidas.",
                    "Los CSV originales no se modifican.",
                ],
            }
            # Publicar archivos solo después de terminar y comprobar los conteos.
            temporary_report = temporary_dir / QUALITY_REPORT_NAME
            temporary_report.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temporary_csv.replace(output_path)
            temporary_report.replace(output_dir / QUALITY_REPORT_NAME)
        finally:
            connection.close()
    return output_path, report


def register_mlflow(
    output_path: Path,
    report: dict,
    raw_dir: Path,
    experiment_name: str,
    tracking_uri: str | None = None,
    artifact_dir: Path | None = None,
) -> str:
    """Carga únicamente el agregado y el catálogo pequeño para registrar las entradas."""
    tracking_uri = (
        tracking_uri
        or f"sqlite:///{(PROJ_ROOT / 'data/mlflow.db').resolve().as_posix()}"
    )
    artifact_dir = (artifact_dir or PROJ_ROOT / "data/mlartifacts").resolve()
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name, artifact_location=artifact_dir.as_uri()
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    catalog_path = raw_dir / CATALOG_NAME
    prepared = pd.read_csv(
        output_path,
        dtype={"fecha": "string", "estacion_id": "int64", "salidas": "int64"},
    )
    catalog = pd.read_csv(catalog_path, encoding=csv_encoding(catalog_path))
    with mlflow.start_run(run_name="prepare-mibici-demand-streaming") as run:
        mlflow.log_input(
            mlflow.data.from_pandas(
                prepared,
                source=str(output_path.resolve()),
                name="mibici_salidas_estacion_fecha",
            ),
            context="prepared",
        )
        mlflow.log_input(
            mlflow.data.from_pandas(
                catalog,
                source=str(catalog_path.resolve()),
                name="mibici_estaciones_2026_09",
            ),
            context="source",
        )
        mlflow.log_params(
            {
                "trip_source_file_count": report["input_files"],
                "chunk_size": report["chunk_size"],
                "processing_mode": report["processing_mode"],
                "deduplication_key": "month,Viaje_Id",
                "filter_version": report["filter_version"],
                "short_trip_limit_seconds": report["short_trip_limit_seconds"],
                "period_start": report["period_start"],
                "period_end": report["period_end"],
                "station_catalog": CATALOG_NAME,
                "station_catalog_url": CATALOG_URL,
                "output_sha256": file_sha256(output_path),
                "catalog_sha256": file_sha256(catalog_path),
            }
        )
        mlflow.log_metrics(
            {
                "trip_rows_read": report["rows_read"],
                "trip_rows_kept": report["rows_kept"],
                "trip_rows_removed": report["rows_removed_total"],
                "daily_station_rows": report["output_rows"],
                "station_rows": len(catalog),
                **{
                    f"removed_{key}": count
                    for key, count in report["rows_removed"].items()
                },
            }
        )
        mlflow.log_artifact(str(output_path), artifact_path="prepared")
        mlflow.log_artifact(
            str(output_path.parent / QUALITY_REPORT_NAME), artifact_path="prepared"
        )
        mlflow.log_artifact(str(catalog_path), artifact_path="source")
        mlflow.log_artifact(str(Path(__file__).resolve()), artifact_path="source")
        mlflow.log_artifact(
            str(PROJ_ROOT / "MiBici/prepare_data.py"), artifact_path="source"
        )
        mlflow.log_artifacts(
            str(PROJ_ROOT / "MiBici/data_cleaning_stages"),
            artifact_path="source/data_cleaning_stages",
        )
        manifest = raw_dir / MANIFEST_NAME
        if manifest.is_file():
            mlflow.log_artifact(str(manifest), artifact_path="source")
        mlflow.log_artifact(
            str(PROJ_ROOT / "references/mibici_urls_2014_2026.txt"),
            artifact_path="source",
        )
        return run.info.run_id


@app.command()
def run(
    raw_dir: Annotated[
        Path, typer.Option(help="Carpeta de CSV originales y catálogo.")
    ] = RAW_DATA_DIR / "mibici",
    output_dir: Annotated[
        Path,
        typer.Option(
            help="Carpeta de resultados; separada de la prueba de cinco meses."
        ),
    ] = PROCESSED_DATA_DIR / "historico",
    experiment_name: Annotated[
        str, typer.Option(help="Experimento de MLflow.")
    ] = "mibici-fase1-historico",
    chunk_size: Annotated[
        int, typer.Option(min=1, help="Máximo de filas de viajes por bloque.")
    ] = 50_000,
) -> None:
    """Procesa CSV mensuales con deduplicación en disco y registra el resultado en MLflow."""
    raw_dir, output_dir = raw_dir.resolve(), output_dir.resolve()
    catalog = raw_dir / CATALOG_NAME
    if not catalog.is_file():
        raise typer.BadParameter(f"No se encontró el catálogo: {catalog}")
    files = sorted(raw_dir.glob("datos_abiertos_*.csv"))
    if not files:
        raise typer.BadParameter(f"No se encontraron CSV de viajes en {raw_dir}")
    logger.info(
        "Preparación de {} archivos; bloques de {} filas; salida {}",
        len(files),
        chunk_size,
        output_dir,
    )
    logger.info(
        "SQLite temporal utiliza disco; al finalizar se conserva únicamente el agregado y el reporte."
    )
    output_path, report = process_files(files, output_dir, chunk_size)
    logger.info(
        "Registro en MLflow del agregado ({} filas), no de los viajes originales.",
        report["output_rows"],
    )
    run_id = register_mlflow(output_path, report, raw_dir, experiment_name)
    logger.success(
        "Preparación terminada: {} leídos; {} retenidos; {} excluidos; {} filas agregadas.",
        report["rows_read"],
        report["rows_kept"],
        report["rows_removed_total"],
        report["output_rows"],
    )
    logger.info(
        "Experimento: {}; run_id: {}; salida: {}", experiment_name, run_id, output_path
    )


if __name__ == "__main__":
    app()
