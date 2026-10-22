"""Limpia los viajes de MiBici y cuenta salidas diarias por estación."""

from datetime import UTC, datetime
import json
import os
from pathlib import Path
import re
from typing import Annotated

from loguru import logger
import mlflow
import pandas as pd
import typer

from MiBici.config import PROCESSED_DATA_DIR, PROJ_ROOT, RAW_DATA_DIR

app = typer.Typer(help="Prepara los datos de MiBici para el análisis de salidas diarias.")
TRIP_COLUMNS = ["Viaje_Id", "Inicio_del_viaje", "Origen_Id"]
OUTPUT_NAME = "mibici_salidas_estacion_fecha.csv"
REPORT_NAME = "mibici_quality_report.json"


def file_period(path: Path) -> str:
    """Obtiene el periodo YYYY-MM del nombre del archivo mensual."""
    # El periodo se usa para comprobar que cada viaje pertenezca al archivo esperado.
    match = re.search(r"datos_abiertos_(\d{4})_(\d{2})", path.name)
    if not match:
        raise ValueError(f"No se pudo identificar el periodo de {path.name}")
    return f"{match.group(1)}-{match.group(2)}"


def csv_encoding(path: Path) -> str:
    """Detecta si el archivo usa UTF-8 o una codificación común de Windows."""
    # Algunos CSV antiguos usan una codificación distinta de UTF-8.
    with path.open("rb") as file:
        sample = file.read(128_000)
    try:
        sample.decode("utf-8-sig")
        return "utf-8-sig"
    except UnicodeDecodeError:
        return "cp1252"


def parse_start_times(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Convierte fechas ISO y fechas con día, mes y año separados por /."""
    # Los archivos tienen fechas ISO y, en algunos meses, día/mes/año.
    text = values.str.strip()
    iso_format = text.str.match(r"^\d{4}-\d{2}-\d{2}", na=False)
    slash_format = text.str.match(r"^\d{1,2}/\d{1,2}/\d{4}", na=False)
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    parsed.loc[iso_format] = pd.to_datetime(text.loc[iso_format], errors="coerce", format="mixed")
    parsed.loc[slash_format] = pd.to_datetime(
        text.loc[slash_format], errors="coerce", format="mixed", dayfirst=True
    )
    formats = pd.Series("unrecognized", index=values.index, dtype="string")
    formats.loc[iso_format] = "year_month_day"
    formats.loc[slash_format] = "day_month_year"
    return parsed, formats


def prepare(raw_dir: Path, output_dir: Path) -> dict:
    """Prepara el conteo diario por estación y genera un reporte de calidad."""
    # Se procesan los CSV mensuales para no reunir todos los viajes crudos en un DF.
    files = sorted(raw_dir.glob("datos_abiertos_*.csv"))
    if not files:
        raise typer.BadParameter(f"No se encontraron archivos mensuales en {raw_dir}")

    period_start = file_period(files[0])
    period_end = file_period(files[-1])
    logger.info("ETAPA 1/3: Lectura de archivos mensuales de MiBici")
    logger.info(
        "Se encontraron {} archivos entre {} y {} en {}",
        len(files),
        period_start,
        period_end,
        raw_dir,
    )

    monthly_results = []
    report = {
        "source": "MiBici Open Data",
        "source_page": "https://mibici.net/en/open-data/",
        "period_start": period_start,
        "period_end": period_end,
        "input_files": len(files),
        "rows_read": 0,
        "rows_kept": 0,
        "rows_removed": {},
        "duplicate_trip_ids": 0,
        "timestamp_formats": {"year_month_day": 0, "day_month_year": 0},
        "notes": [
            "Se utilizaron Viaje_Id, Inicio_del_viaje y Origen_Id; se omitieron identificadores de usuario y datos demográficos.",
            "Los duplicados se eliminan dentro de cada archivo mensual; los archivos se validan contra el mes indicado en su nombre.",
            "Los picos de demanda se conservan porque no hay una regla validada que permita clasificarlos como errores.",
            "No se generan ceros para estación-fecha sin viajes observados; el nombre de la estación no se necesita para este análisis.",
        ],
    }
    removed = {
        "missing_trip_id": 0,
        "invalid_timestamp": 0,
        "invalid_station_id": 0,
        "timestamp_outside_file_month": 0,
    }
    inventory = []
    progress_interval = max(1, (len(files) + 9) // 10)

    for index, path in enumerate(files, start=1):
        period = file_period(path)
        inventory.append(
            {
                "filename": path.name,
                "bytes": path.stat().st_size,
            }
        )

        # Solo se leen ID, fecha de inicio y estación de origen para este objetivo.
        month_data = pd.read_csv(
            path,
            usecols=TRIP_COLUMNS,
            dtype={column: "string" for column in TRIP_COLUMNS},
            encoding=csv_encoding(path),
        )
        report["rows_read"] += len(month_data)
        trip_id = month_data["Viaje_Id"].str.strip()
        start, timestamp_formats = parse_start_times(month_data["Inicio_del_viaje"])
        report["timestamp_formats"]["year_month_day"] += int(
            timestamp_formats.eq("year_month_day").sum()
        )
        report["timestamp_formats"]["day_month_year"] += int(
            timestamp_formats.eq("day_month_year").sum()
        )
        station_number = pd.to_numeric(month_data["Origen_Id"].str.strip(), errors="coerce")

        # Se validan los campos usados; los datos demográficos no afectan el conteo.
        valid_id = trip_id.notna() & trip_id.ne("")
        removed["missing_trip_id"] += int((~valid_id).sum())
        valid_time = start.notna()
        removed["invalid_timestamp"] += int((valid_id & ~valid_time).sum())
        valid_station = station_number.notna() & station_number.gt(0)
        valid_station &= station_number.mod(1).eq(0)
        removed["invalid_station_id"] += int((valid_id & valid_time & ~valid_station).sum())
        year, month = map(int, period.split("-"))
        in_period = start.dt.year.eq(year) & start.dt.month.eq(month)
        removed["timestamp_outside_file_month"] += int(
            (valid_id & valid_time & valid_station & ~in_period).sum()
        )

        # Se descartan filas inválidas, fuera de mes o sin estación de origen válida.
        keep = valid_id & valid_time & valid_station & in_period
        cleaned = pd.DataFrame(
            {
                "trip_id": trip_id[keep].astype(str),
                "fecha": start[keep].dt.floor("D"),
                "estacion_id": station_number[keep].astype("int64"),
            }
        )
        report["duplicate_trip_ids"] += int(cleaned["trip_id"].duplicated().sum())
        cleaned = cleaned.drop_duplicates("trip_id", keep="first")
        report["rows_kept"] += len(cleaned)
        # Cada viaje cuenta como una salida de su estación en la fecha de inicio.
        monthly_results.append(
            cleaned.groupby(["fecha", "estacion_id"], as_index=False)
            .size()
            .rename(columns={"size": "salidas"})
        )

        if index % progress_interval == 0 or index == len(files):
            logger.info(
                "Progreso: {}/{} archivos; {} viajes leídos y {} válidos hasta ahora",
                index,
                len(files),
                report["rows_read"],
                report["rows_kept"],
            )

    report["rows_removed"] = removed | {"duplicate_trip_ids": report["duplicate_trip_ids"]}
    report["rows_removed_total"] = report["rows_read"] - report["rows_kept"]
    report["files"] = inventory

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_NAME
    quality_path = output_dir / REPORT_NAME
    # Se combinan las agregaciones mensuales en el DataFrame final.
    prepared = pd.concat(monthly_results, ignore_index=True)
    if prepared.empty:
        raise ValueError("No quedaron viajes válidos; no se generó el dataset.")
    prepared = prepared.sort_values(["fecha", "estacion_id"]).reset_index(drop=True)
    prepared.to_csv(output_path, index=False, encoding="utf-8")
    report["output_rows"] = len(prepared)
    removal_labels = {
        "missing_trip_id": "Viaje sin identificador",
        "invalid_timestamp": "Fecha u hora inválida",
        "invalid_station_id": "Estación de origen inválida",
        "timestamp_outside_file_month": "Fecha fuera del mes del archivo",
        "duplicate_trip_ids": "Viaje duplicado",
    }
    report["executive_summary"] = {
        "period": f"{period_start} a {period_end}",
        "input_files": len(files),
        "rows_read": report["rows_read"],
        "valid_trips_kept": report["rows_kept"],
        "rows_removed_total": report["rows_removed_total"],
        "rows_removed_by_reason": {
            removal_labels[reason]: count
            for reason, count in report["rows_removed"].items()
        },
        "daily_station_records": report["output_rows"],
        "output_csv": str(output_path),
    }
    report["process_steps"] = [
        {
            "step": 1,
            "activity": "Lectura de los CSV mensuales",
            "result": f"{len(files)} archivos entre {period_start} y {period_end}",
        },
        {
            "step": 2,
            "activity": "Validación y limpieza de viajes",
            "result": (
                f"{report['rows_kept']} viajes válidos conservados; "
                f"{report['rows_removed_total']} filas descartadas"
            ),
        },
        {
            "step": 3,
            "activity": "Conteo diario por estación",
            "result": f"{report['output_rows']} registros agregados en el CSV preparado",
        },
    ]
    report["created_at_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    quality_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("ETAPA 2/3: Validación, limpieza y conteo de salidas completados")
    logger.info(
        "Resultado: {} viajes leídos; {} conservados; {} descartados; {} registros diarios por estación",
        report["rows_read"],
        report["rows_kept"],
        report["rows_removed_total"],
        report["output_rows"],
    )
    for reason, count in report["executive_summary"]["rows_removed_by_reason"].items():
        if count:
            logger.info("  - {}: {}", reason, count)
    logger.info("Dataset preparado: {}", output_path)
    logger.info("Reporte de calidad: {}", quality_path)
    return report


@app.command()
def run(
    raw_dir: Annotated[
        Path, typer.Option(help="Carpeta con los CSV originales de MiBici.")
    ] = RAW_DATA_DIR / "mibici",
    output_dir: Annotated[
        Path, typer.Option(help="Carpeta para el dataset limpio y el reporte.")
    ] = PROCESSED_DATA_DIR,
    experiment_name: Annotated[
        str, typer.Option(help="Nombre del experimento en MLflow.")
    ] = "mibici-data-preparation",
) -> None:
    """Prepara los datos y registra el resultado de la ejecución en MLflow."""
    logger.info("Iniciando preparación de datos y registro en MLflow")
    report = prepare(raw_dir, output_dir)
    # MLflow guarda metadatos en SQLite y los archivos registrados como artefactos.
    tracking_db = (PROJ_ROOT / "data" / "mlflow.db").resolve()
    default_tracking_uri = f"sqlite:///{tracking_db}"
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", default_tracking_uri))
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        artifact_location = (PROJ_ROOT / "data" / "mlartifacts").resolve().as_uri()
        experiment_id = mlflow.create_experiment(
            experiment_name, artifact_location=artifact_location
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    logger.info("ETAPA 3/3: Registro de métricas y archivos en MLflow")
    with mlflow.start_run(run_name="prepare-mibici-daily-departures") as active_run:
        mlflow.log_param("source", report["source"])
        mlflow.log_param("period", f"{report['period_start']} to {report['period_end']}")
        mlflow.log_param("input_file_count", report["input_files"])
        mlflow.log_param("cleaning_script_version", "1")
        mlflow.log_metric("rows_read", report["rows_read"])
        mlflow.log_metric("rows_kept", report["rows_kept"])
        mlflow.log_metric("rows_removed", report["rows_removed_total"])
        mlflow.log_metric("duplicate_trip_ids", report["duplicate_trip_ids"])
        mlflow.log_metric("output_rows", report["output_rows"])
        mlflow.log_artifact(str(output_dir / OUTPUT_NAME), artifact_path="dataset")
        mlflow.log_artifact(str(output_dir / REPORT_NAME), artifact_path="quality")
        mlflow.set_tag("data_source", report["source_page"])
    logger.success("Ejecución registrada en MLflow")
    logger.info("Experimento: {} (ID: {})", experiment_name, experiment_id)
    logger.info("ID de ejecución: {}", active_run.info.run_id)
    logger.info("Base local de MLflow: {}", tracking_db)
    logger.info("Artefactos locales de MLflow: {}", PROJ_ROOT / "data" / "mlartifacts")
    logger.success(
        "Proceso terminado: {} filas de salida para el periodo {} a {}",
        report["output_rows"],
        report["period_start"],
        report["period_end"],
    )


if __name__ == "__main__":
    app()
