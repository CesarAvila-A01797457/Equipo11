"""Preprocesamiento de MiBici: calendario, split temporal y transformaciones de train.

Conserva únicamente los estación-días presentes. No imputa ceros, añade rezagos
ni entrena un modelo. Usa matrices dispersas para evitar one-hot denso en RAM.
"""

import argparse
import hashlib
import json
import platform
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CATEGORICAL = ["estacion_id", "dia_semana", "mes"]
NUMERIC = ["anio"]
PREDICTORS = CATEGORICAL + NUMERIC


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    Path(path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def load_daily_counts(input_path, quality_report=None):
    data = pd.read_csv(
        input_path, usecols=["fecha", "estacion_id", "salidas"], dtype="string"
    )
    if data.empty or data.isna().any().any():
        raise ValueError("El agregado está vacío o tiene campos obligatorios faltantes")
    data["fecha"] = pd.to_datetime(data.fecha, format="%Y-%m-%d", errors="coerce")
    if data.fecha.isna().any():
        raise ValueError("Hay fechas inválidas")
    for column, minimum in [("estacion_id", 1), ("salidas", 0)]:
        numeric = pd.to_numeric(data[column], errors="coerce")
        if (numeric.isna() | numeric.lt(minimum) | numeric.mod(1).ne(0)).any():
            raise ValueError(f"Hay valores inválidos en {column}")
        data[column] = numeric.astype("int64")
    if data.duplicated(["fecha", "estacion_id"]).any():
        raise ValueError("Hay claves estación-fecha duplicadas")
    if quality_report is not None:
        quality = json.loads(Path(quality_report).read_text(encoding="utf-8-sig"))
        if quality["rows_kept"] != int(data.salidas.sum()) or quality[
            "output_rows"
        ] != len(data):
            raise ValueError(
                "El reporte de calidad no coincide con el dataset de entrada"
            )
    return data.sort_values(["fecha", "estacion_id"]).reset_index(drop=True)


def build_calendar_features(data):
    frame = data.copy()
    frame["estacion_id"] = frame.estacion_id.astype("string")
    frame["dia_semana"] = frame.fecha.dt.day_of_week.astype("string")
    frame["mes"] = frame.fecha.dt.month.astype("string")
    frame["anio"] = frame.fecha.dt.year.astype("int64")
    return frame


def temporal_split(frame, train_end, validation_end):
    train_end = pd.Timestamp(datetime.strptime(train_end, "%Y-%m-%d"))
    validation_end = pd.Timestamp(datetime.strptime(validation_end, "%Y-%m-%d"))
    if train_end >= validation_end:
        raise ValueError("train_end debe ser anterior a validation_end")
    parts = {
        "train": frame.loc[frame.fecha.le(train_end)].copy(),
        "validation": frame.loc[
            frame.fecha.gt(train_end) & frame.fecha.le(validation_end)
        ].copy(),
        "test": frame.loc[frame.fecha.gt(validation_end)].copy(),
    }
    if any(part.empty for part in parts.values()):
        raise ValueError(
            "Entrenamiento, validación y prueba deben contener registros; revise los cortes"
        )
    if sum(map(len, parts.values())) != len(frame):
        raise RuntimeError("Los splits no conservan todas las filas")
    if not (
        parts["train"].fecha.max() < parts["validation"].fecha.min()
        and parts["validation"].fecha.max() < parts["test"].fecha.min()
    ):
        raise RuntimeError("Hay solapamiento temporal")
    return parts


def fit_preprocessor(train):
    transformer = ColumnTransformer(
        [
            (
                "categoria",
                OneHotEncoder(
                    handle_unknown="ignore", sparse_output=True, dtype=np.float32
                ),
                CATEGORICAL,
            ),
            ("numero", StandardScaler(), NUMERIC),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )
    # No se incluyen salidas ni fecha sin transformar entre los predictores.
    transformer.fit(train[PREDICTORS])
    return transformer


def save_transformed(parts, transformer, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(transformer, output_dir / "preprocessor.joblib")
    names = transformer.get_feature_names_out().tolist()
    write_json(output_dir / "columnas_transformadas.json", names)
    known_stations = set(transformer.named_transformers_["categoria"].categories_[0])
    summaries = {}
    for name, part in parts.items():
        print(f"Transformando {name}: {len(part):,} filas...", flush=True)
        matrix = sparse.csr_matrix(
            transformer.transform(part[PREDICTORS]), dtype=np.float32
        )
        target = part.salidas.to_numpy(dtype=np.int64)
        if (
            matrix.shape != (len(part), len(names))
            or not np.isfinite(matrix.data).all()
        ):
            raise RuntimeError(f"Matriz inválida en {name}")
        part.to_csv(output_dir / f"{name}.csv", index=False)
        sparse.save_npz(output_dir / f"X_{name}.npz", matrix)
        np.save(output_dir / f"y_{name}.npy", target)
        unknown = ~part.estacion_id.isin(known_stations)
        summaries[name] = {
            "rows": len(part),
            "date_start": part.fecha.min().strftime("%Y-%m-%d"),
            "date_end": part.fecha.max().strftime("%Y-%m-%d"),
            "departures": int(target.sum()),
            "stations": int(part.estacion_id.nunique()),
            "stations_unseen_in_training": int(
                part.loc[unknown, "estacion_id"].nunique()
            ),
            "rows_unseen_station": int(unknown.sum()),
            "matrix_shape": list(matrix.shape),
            "matrix_nonzero_values": int(matrix.nnz),
            "matrix_storage_mb": round(
                (matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)
                / 1024**2,
                3,
            ),
        }
        del matrix, target
    return summaries


def log_preprocessing(
    parts,
    input_path,
    output_dir,
    report,
    experiment_name,
    source_run_id=None,
    tracking_uri=None,
    artifact_dir=None,
):
    import mlflow
    import mlflow.data

    tracking_uri = (
        tracking_uri or f"sqlite:///{(ROOT / 'data/mlflow.db').resolve().as_posix()}"
    )
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    if source_run_id:
        source = client.get_run(source_run_id)
        expected = source.data.params.get("output_sha256")
        if expected and expected != report["input_sha256"]:
            raise ValueError(
                "La huella del dataset no coincide con el run de preparación"
            )
    if mlflow.get_experiment_by_name(experiment_name) is None:
        location = (artifact_dir or ROOT / "data/mlartifacts").resolve().as_uri()
        mlflow.create_experiment(experiment_name, artifact_location=location)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="preprocesamiento-calendario-mibici") as run:
        mlflow.set_tags(
            {
                "stage": "preprocessing",
                "dataset_version": "calendar_onehot_v1",
                "preparation_run_id": source_run_id or "not_provided",
                "fit_scope": "train_only",
            }
        )
        mlflow.log_params(
            {
                "input_sha256": report["input_sha256"],
                "train_end": report["train_end"],
                "validation_end": report["validation_end"],
                "predictors": ",".join(PREDICTORS),
                "categorical_encoding": "OneHotEncoder(handle_unknown=ignore)",
                "numeric_scaling": "StandardScaler(anio)",
                "zero_imputation": False,
                "target_scaled": False,
                "code_sha256": sha256(Path(__file__)),
            }
        )
        for name, part in parts.items():
            mlflow.log_input(
                mlflow.data.from_pandas(
                    part,
                    source=str((Path(output_dir) / f"{name}.csv").resolve()),
                    name=f"mibici_calendar_{name}",
                    targets="salidas",
                ),
                context=name,
            )
            info = report["splits"][name]
            mlflow.log_metrics(
                {
                    f"rows_{name}": info["rows"],
                    f"departures_{name}": info["departures"],
                    f"unknown_stations_{name}": info["stations_unseen_in_training"],
                }
            )
        mlflow.log_metric("transformed_columns", report["transformed_columns"])
        # Conservar CSV, matrices, etiquetas y transformador, no solo metadatos.
        for path in sorted(Path(output_dir).iterdir()):
            if path.is_file() and path.name != "mlflow_preprocessing_run.json":
                mlflow.log_artifact(str(path), artifact_path="preprocessing")
        mlflow.log_artifact(str(Path(__file__).resolve()), artifact_path="source")
        return run.info.run_id


def run_preprocessing(
    input_path,
    output_dir,
    quality_report=None,
    train_end="2025-12-31",
    validation_end="2026-06-30",
    experiment_name="mibici-fase1-historico",
    source_run_id=None,
    log_mlflow=False,
    tracking_uri=None,
):
    input_path, output_dir = Path(input_path).resolve(), Path(output_dir).resolve()
    if output_dir == input_path.parent or input_path.is_relative_to(output_dir):
        raise ValueError("Use una carpeta de salida separada del CSV de entrada")
    print("Validando datos y construyendo calendario...", flush=True)
    data = load_daily_counts(input_path, quality_report)
    frame = build_calendar_features(data)
    del data
    parts = temporal_split(frame, train_end, validation_end)
    rows, departures = len(frame), int(frame.salidas.sum())
    del frame
    print(
        "Ajustando codificador y escala exclusivamente con entrenamiento...", flush=True
    )
    transformer = fit_preprocessor(parts["train"])
    summaries = save_transformed(parts, transformer, output_dir)
    report = {
        "dataset_version": "calendar_onehot_v1",
        "input_sha256": sha256(input_path),
        "input_rows": rows,
        "input_departures": departures,
        "rows_removed": 0,
        "train_end": train_end,
        "validation_end": validation_end,
        "source_preparation_run_id": source_run_id,
        "splits": summaries,
        "predictors_before_encoding": PREDICTORS,
        "target": "salidas",
        "transformed_columns": len(transformer.get_feature_names_out()),
        "fit_scope": "train_only",
        "scaled_numeric_features": NUMERIC,
        "training_year_mean": float(transformer.named_transformers_["numero"].mean_[0]),
        "training_year_scale": float(
            transformer.named_transformers_["numero"].scale_[0]
        ),
        "target_scaled": False,
        "zero_imputation": False,
        "lags_added": False,
        "rows_conserved": rows == sum(s["rows"] for s in summaries.values()),
        "departures_conserved": departures
        == sum(s["departures"] for s in summaries.values()),
        "software_versions": {
            name: version(name)
            for name in ["pandas", "numpy", "scipy", "scikit-learn", "joblib"]
        },
        "python": platform.python_version(),
        "limitations": [
            "Solo se representan estación-días presentes; ausencia de fila no equivale a cero.",
            "No se entrena ni evalúa un modelo predictivo en esta etapa.",
            "Estaciones no vistas en train tienen ceros en su grupo one-hot; se reportan por separado.",
            "No se imputa: el agregado validado no tiene campos obligatorios faltantes.",
            "No se aplica PCA: se conserva identidad categórica y representación dispersa.",
            "El año representa tendencia; no supone que esa tendencia se extrapole correctamente.",
            "El calendario es conocido de antemano; no se requieren observaciones futuras para generarlo.",
        ],
    }
    if not report["rows_conserved"] or not report["departures_conserved"]:
        raise RuntimeError("No se conservan las filas o salidas")
    write_json(output_dir / "reporte_preprocesamiento.json", report)
    manifest = {
        p.name: sha256(p)
        for p in sorted(output_dir.iterdir())
        if p.is_file()
        and p.name not in {"manifest_sha256.json", "mlflow_preprocessing_run.json"}
    }
    write_json(output_dir / "manifest_sha256.json", manifest)
    if log_mlflow:
        run_id = log_preprocessing(
            parts,
            input_path,
            output_dir,
            report,
            experiment_name,
            source_run_id,
            tracking_uri,
        )
        write_json(
            output_dir / "mlflow_preprocessing_run.json",
            {
                "run_id": run_id,
                "experiment_name": experiment_name,
                "source_preparation_run_id": source_run_id,
            },
        )
        report["preprocessing_run_id"] = run_id
    print(
        f"Preprocesamiento terminado: {rows:,} filas conservadas; {report['transformed_columns']} columnas transformadas.",
        flush=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-path",
        type=Path,
        default=ROOT / "data/processed/historico/mibici_salidas_estacion_fecha.csv",
    )
    parser.add_argument(
        "--quality-report",
        type=Path,
        default=ROOT / "data/processed/historico/mibici_quality_report.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/processed/preprocesado"
    )
    parser.add_argument("--train-end", default="2025-12-31")
    parser.add_argument("--validation-end", default="2026-06-30")
    parser.add_argument("--experiment-name", default="mibici-fase1-historico")
    parser.add_argument("--source-run-id")
    parser.add_argument("--tracking-uri")
    parser.add_argument("--no-mlflow", action="store_true")
    options = vars(parser.parse_args())
    options["log_mlflow"] = not options.pop("no_mlflow")
    print(
        json.dumps(
            run_preprocessing(**options), indent=2, ensure_ascii=False, allow_nan=False
        )
    )


if __name__ == "__main__":
    main()
