"""Simula fallas de identificación de estaciones y mide drift y desempeño."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from MiBici.features import (
    PREDICTORS, build_calendar_features, load_daily_counts, sha256, temporal_split,
)
from MiBici.modeling.train import calculate_metrics


def corrupt_station_ids(frame, known_stations, fraction, seed):
    """Cambia IDs conocidos por IDs sintéticos, manteniendo filas y objetivos."""
    if not 0 <= fraction <= 1:
        raise ValueError("La fracción debe estar entre cero y uno")
    result = frame.copy(deep=True)
    eligible = np.flatnonzero(frame.estacion_id.isin(known_stations).to_numpy())
    order = np.random.default_rng(seed).permutation(eligible)
    selected = order[:int(len(eligible) * fraction)]
    # Una correspondencia uno a uno conserva las claves de estación y fecha.
    offset = int(frame.estacion_id.astype("int64").max()) + 1
    mapping = {
        station: str(offset + index)
        for index, station in enumerate(sorted(set(known_stations)))
    }
    column = result.columns.get_loc("estacion_id")
    result.iloc[selected, column] = result.iloc[selected, column].map(mapping)
    return result, len(selected)


def categorical_distance(reference, current):
    """Distancia de variación total, entre cero y uno, para dos categorías."""
    left = reference.value_counts(normalize=True)
    right = current.value_counts(normalize=True)
    categories = left.index.union(right.index)
    return float(
        (left.reindex(categories, fill_value=0)
         - right.reindex(categories, fill_value=0)).abs().sum() / 2
    )


def evaluate_scenarios(model, validation, fractions, seed, drift_threshold, mae_threshold):
    """Evalúa el modelo congelado sobre las mismas filas y objetivos."""
    if validation.empty:
        raise ValueError("La validación está vacía")
    if drift_threshold <= 0 or mae_threshold <= 0:
        raise ValueError("Los umbrales deben ser positivos")
    encoder = model.named_steps["preprocessor"].named_transformers_["categoria"]
    known_stations = set(encoder.categories_[0])
    base = calculate_metrics(
        validation.salidas, model.predict(validation[PREDICTORS])
    )
    if base["mae"] == 0:
        raise ValueError("El MAE base es cero y no permite calcular cambio relativo")
    rows = []
    for fraction in [0.0, *fractions]:
        current, changed = corrupt_station_ids(
            validation, known_stations, fraction, seed
        )
        metrics = calculate_metrics(
            current.salidas, model.predict(current[PREDICTORS])
        )
        distance = categorical_distance(validation.estacion_id, current.estacion_id)
        increase = metrics["mae"] / base["mae"] - 1
        drift_alert = distance >= drift_threshold
        performance_alert = increase >= mae_threshold
        rows.append({
            "fraction_of_known_rows": fraction,
            "changed_rows": changed,
            "total_rows": len(current),
            "unknown_station_rate": float(
                (~current.estacion_id.isin(known_stations)).mean()
            ),
            "station_total_variation": distance,
            "mae": metrics["mae"],
            "bias": metrics["bias"],
            "mae_relative_increase": increase,
            "drift_alert": bool(drift_alert),
            "performance_alert": bool(performance_alert),
            "joint_alert": bool(drift_alert and performance_alert),
        })
    return rows


def save_outputs(output_dir, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(report["scenarios"])
    frame.to_csv(output_dir / "comparacion.csv", index=False)
    (output_dir / "reporte_drift.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    x = frame.fraction_of_known_rows.to_numpy() * 100
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    axes[0].plot(x, frame.mae, "o-", label="MAE")
    axes[0].axhline(
        frame.mae.iloc[0] * (1 + report["thresholds"]["mae_relative_increase"]),
        color="red", linestyle="--", label="Umbral de desempeño",
    )
    axes[0].set_ylabel("MAE en salidas por estación y día")
    axes[1].plot(x, frame.station_total_variation, "o-", label="Distancia TV")
    axes[1].axhline(
        report["thresholds"]["station_total_variation"],
        color="red", linestyle="--", label="Umbral de drift",
    )
    axes[1].set_ylabel("Distancia respecto a validación original")
    for axis in axes:
        axis.set_xlabel("Filas con ID conocido alteradas (%)")
        axis.legend()
        axis.grid(alpha=0.25)
    fig.savefig(output_dir / "drift_y_desempeno.png", dpi=160)
    plt.close(fig)
    lines = [
        "# Simulación de drift de identificadores de estación", "",
        "Se evalúa un fallo sintético del pipeline de entrada que cambia IDs de "
        "estación por códigos desconocidos para el modelo. Se mantienen las fechas, "
        "los conteos reales y el orden de las observaciones. El modelo queda congelado.",
        "",
        "La referencia usa validación de enero a junio de 2026. Los escenarios "
        "alteran subconjuntos anidados con semilla fija. La fracción se aplica solo "
        "a las filas cuya estación es conocida por el preprocesador.",
        "",
        "| Fracción alterada | Filas alteradas | TV | MAE | Sesgo | Cambio MAE | Alerta conjunta |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["scenarios"]:
        lines.append(
            f"| {row['fraction_of_known_rows']:.0%} | {row['changed_rows']} "
            f"| {row['station_total_variation']:.3f} | {row['mae']:.4f} "
            f"| {row['bias']:+.4f} | {row['mae_relative_increase']:+.1%} "
            f"| {row['joint_alert']} |"
        )
    lines += [
        "",
        "Se propone alertar con TV >= "
        f"{report['thresholds']['station_total_variation']:.2f} y con un incremento "
        f"relativo del MAE >= {report['thresholds']['mae_relative_increase']:.0%}. "
        "Son umbrales educativos definidos antes de ejecutar la simulación. "
        "La alerta conjunta requiere que se cumplan ambos criterios.",
        "",
        "Ante drift se revisaría el catálogo y la correspondencia de IDs. Ante "
        "degradación se investigaría su causa. Una vez corregidas las entradas, "
        "se volvería a evaluar. El reentrenamiento se consideraría si el cambio "
        "persistiera con datos válidos y nuevas etiquetas.",
        "",
        "El drift de entrada se puede medir sin etiquetas. La pérdida de desempeño "
        "requiere conteos observados. Esta prueba representa corrupción de una "
        "variable de entrada y no permite concluir que exista drift real en MiBici "
        "ni que los IDs sintéticos representen estaciones nuevas.",
        "",
        "La comparación corresponde a validación y es independiente de las "
        "métricas de prueba publicadas en la Fase 2.",
    ]
    (output_dir / "DRIFT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    import mlflow
    import mlflow.sklearn
    from mlflow.models import Model

    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=root / "models/mibici-poisson-v2")
    parser.add_argument("--input-path", type=Path, default=root / "data/processed/historico_filtrado/mibici_salidas_estacion_fecha.csv")
    parser.add_argument("--quality-report", type=Path, default=root / "data/processed/historico_filtrado/mibici_quality_report.json")
    parser.add_argument("--output-dir", type=Path, default=root / "reports/fase3/drift")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-name", default="mibici-fase3-drift")
    args = parser.parse_args()

    data = load_daily_counts(args.input_path, args.quality_report)
    parts = temporal_split(build_calendar_features(data), "2025-12-31", "2026-06-30")
    model = mlflow.sklearn.load_model(str(args.model_path))
    scenarios = evaluate_scenarios(model, parts["validation"], [0.1, 0.3, 0.6, 1.0], args.seed, 0.1, 0.1)
    metadata = Model.load(str(args.model_path / "MLmodel"))
    report = {
        "scenario": "synthetic_station_id_corruption",
        "model_path": str(args.model_path),
        "model_pickle_sha256": sha256(args.model_path / "model.pkl"),
        "source_model_run_id": metadata.run_id,
        "input_sha256": sha256(args.input_path),
        "evaluation_period": ["2026-01-01", "2026-06-30"],
        "seed": args.seed,
        "thresholds": {"station_total_variation": 0.1, "mae_relative_increase": 0.1},
        "scenarios": scenarios,
    }
    mlflow.set_tracking_uri(f"sqlite:///{(root / 'data/mlflow.db').as_posix()}")
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name="station-id-drift") as run:
        report["drift_run_id"] = run.info.run_id
        mlflow.log_params({
            "seed": args.seed, "scenario": report["scenario"],
            "source_model_run_id": report["source_model_run_id"],
            "input_sha256": report["input_sha256"],
            "model_pickle_sha256": report["model_pickle_sha256"],
            "tv_threshold": 0.1, "relative_mae_threshold": 0.1,
        })
        for step, row in enumerate(scenarios):
            mlflow.log_metrics(
                {key: float(value) for key, value in row.items()}, step=step
            )
        save_outputs(args.output_dir, report)
        mlflow.log_artifacts(str(args.output_dir), artifact_path="drift")
        mlflow.log_artifact(__file__, artifact_path="source")
    print(pd.DataFrame(scenarios).to_string(index=False))
    print(f"MLflow run_id: {report['drift_run_id']}")
    print(f"Reportes: {args.output_dir}")


if __name__ == "__main__":
    main()
