"""Prepara los cortes temporales y evalúa modelos para Fase 2."""

from pathlib import Path

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
import typer

from MiBici.config import PROCESSED_DATA_DIR, PROJ_ROOT, RANDOM_SEED, RAW_DATA_DIR
from MiBici.feature_engineering import (
    FEATURE_SETS,
    NEEDS_CATALOG,
    FeaturePipeline,
    build_steps,
    load_station_catalog,
)
from MiBici.features import (
    PREDICTORS,
    build_calendar_features,
    build_preprocessor,
    load_daily_counts,
    sha256,
    temporal_split,
)

app = typer.Typer()
FILTERED_DIR = PROCESSED_DATA_DIR / "historico_filtrado"
CATALOG_PATH = RAW_DATA_DIR / "mibici" / "nomenclatura_2026_09.csv"
FEATURE_INPUTS = ["fecha", "estacion_id"]
FOREST_CONFIGS = {
    1: {
        "n_estimators": 50,
        "max_depth": 12,
        "min_samples_leaf": 20,
        "max_features": 0.3,
        "max_samples": 0.1,
    },
    2: {
        "n_estimators": 75,
        "max_depth": 16,
        "min_samples_leaf": 20,
        "max_features": 0.3,
        "max_samples": 0.15,
    },
    3: {
        "n_estimators": 100,
        "max_depth": 20,
        "min_samples_leaf": 10,
        "max_features": 0.3,
        "max_samples": 0.2,
    },
}


def build_model_pipeline(model, feature_steps=None):
    """Combina el preprocesamiento y el modelo para entrenar e inferir.

    Sin ``feature_steps`` conserva el pipeline v1 registrado. Con pasos, el
    pipeline recibe ``fecha`` y ``estacion_id`` y deriva las características.
    """
    if feature_steps is None:
        return Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                ("model", model),
            ]
        )
    features = FeaturePipeline(feature_steps)
    return Pipeline(
        [
            ("features", features),
            ("preprocessor", features.build_column_transformer()),
            ("model", model),
        ]
    )


def segmented_metrics(evaluation, predictions, known_stations, prefix):
    """MAE y sesgo totales, por estación conocida/nueva y por mes."""
    actual = evaluation["salidas"].to_numpy()
    metrics = {
        f"{prefix}_{key}": value
        for key, value in calculate_metrics(actual, predictions).items()
    }
    known = evaluation["estacion_id"].isin(known_stations).to_numpy()
    for label, mask in (("known_station", known), ("new_station", ~known)):
        if mask.any():
            part = calculate_metrics(actual[mask], predictions[mask])
            metrics[f"{prefix}_mae_{label}"] = part["mae"]
            metrics[f"{prefix}_rows_{label}"] = float(mask.sum())
    months = evaluation["fecha"].dt.strftime("%Y_%m").to_numpy()
    for month in sorted(set(months)):
        mask = months == month
        metrics[f"{prefix}_mae_{month}"] = calculate_metrics(
            actual[mask], predictions[mask]
        )["mae"]
    return metrics


def calculate_metrics(actual, predictions):
    """Calcula MAE y sesgo; el sesgo positivo indica sobreestimación."""
    actual = np.asarray(actual, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    if actual.ndim != 1 or predictions.ndim != 1 or actual.shape != predictions.shape:
        raise ValueError("actual y predictions deben ser vectores del mismo tamaño")
    if actual.size == 0:
        raise ValueError("No se pueden calcular métricas sin observaciones")
    if not np.isfinite(actual).all() or not np.isfinite(predictions).all():
        raise ValueError("actual y predictions deben contener valores finitos")
    return {
        "mae": float(mean_absolute_error(actual, predictions)),
        "bias": float(np.mean(predictions - actual)),
    }


@app.command()
def main(
    input_path: Path = FILTERED_DIR / "mibici_salidas_estacion_fecha.csv",
    quality_report: Path = FILTERED_DIR / "mibici_quality_report.json",
    train_end: str = "2025-12-31",
    validation_end: str = "2026-06-30",
    poisson: bool = typer.Option(False, help="Evalúa una regresión de Poisson."),
    alpha: float = typer.Option(0.1, min=0.0, help="Regularización de Poisson."),
    random_forest: bool = typer.Option(
        False, help="Evalúa un Random Forest con criterio Poisson."
    ),
    forest_config: int = typer.Option(
        1, min=1, max=3, help="Configuración de Random Forest: 1, 2 o 3."
    ),
    test: bool = typer.Option(
        False, help="Evalúa el modelo elegido en el periodo de prueba."
    ),
    experiment: str | None = typer.Option(
        None, help="Experimento de MLflow para esta corrida."
    ),
    tracking_uri: str | None = typer.Option(None, help="URI de seguimiento de MLflow."),
    register_model: str | None = typer.Option(
        None, help="Nombre en MLflow Model Registry."
    ),
    baseline: bool = typer.Option(
        False, help="Evalúa la media de entrenamiento en validación."
    ),
    feature_set: str = typer.Option(
        "v1", help=f"Conjunto de características: {', '.join(FEATURE_SETS)}."
    ),
    catalog_path: Path = CATALOG_PATH,
):
    """Valida las particiones temporales y evalúa el modelo solicitado."""
    selected_models = sum((baseline, poisson, random_forest))
    if selected_models > 1:
        raise typer.BadParameter(
            "Elige solo uno entre --baseline, --poisson y --random-forest."
        )

    if test and not (baseline or poisson):
        raise typer.BadParameter("--test requiere --baseline o --poisson.")

    if experiment and not selected_models:
        raise typer.BadParameter(
            "--experiment requiere --baseline, --poisson o --random-forest."
        )

    if register_model and not (poisson and test and experiment):
        raise typer.BadParameter(
            "--register-model requiere --poisson, --test y --experiment."
        )

    if random_forest and forest_config not in FOREST_CONFIGS:
        raise typer.BadParameter("--forest-config debe ser 1, 2 o 3.")

    if feature_set not in FEATURE_SETS:
        raise typer.BadParameter(f"--feature-set debe ser uno de: {', '.join(FEATURE_SETS)}")
    catalog = load_station_catalog(catalog_path) if feature_set in NEEDS_CATALOG else None
    # v1 conserva exactamente el pipeline registrado en Fase 2.
    feature_steps = None if feature_set == "v1" else build_steps(feature_set, catalog)
    inputs = PREDICTORS if feature_steps is None else FEATURE_INPUTS

    data = load_daily_counts(input_path, quality_report)
    parts = temporal_split(build_calendar_features(data), train_end, validation_end)
    for name, part in parts.items():
        first_date = part.fecha.min().date()
        last_date = part.fecha.max().date()
        typer.echo(f"{name}: {len(part)} filas ({first_date} a {last_date})")

    if selected_models:
        if poisson:
            model = PoissonRegressor(alpha=alpha, max_iter=500)
            model_label = f"poisson-alpha-{alpha:g}"
        elif random_forest:
            model = RandomForestRegressor(
                criterion="poisson",
                random_state=RANDOM_SEED,
                n_jobs=-1,
                **FOREST_CONFIGS[forest_config],
            )
            model_label = f"random-forest-config-{forest_config}"
        else:
            model = DummyRegressor(strategy="mean")
            model_label = "baseline-mean"
        if feature_set != "v1":
            model_label = f"{model_label}-{feature_set}"
        train = parts["train"]
        split = "test" if test else "validation"
        evaluation = parts[split]
        period = "prueba" if test else "validación"
        pipeline = build_model_pipeline(model, feature_steps)
        pipeline.fit(train[inputs], train["salidas"])
        predictions = pipeline.predict(evaluation[inputs])
        actual = evaluation["salidas"].to_numpy()
        metrics = calculate_metrics(actual, predictions)
        mae, bias = metrics["mae"], metrics["bias"]
        detail = segmented_metrics(
            evaluation, predictions, set(train["estacion_id"]), split
        )
        n_features = len(pipeline.named_steps["preprocessor"].get_feature_names_out())
        if poisson:
            typer.echo(f"Regresión de Poisson (alpha={alpha:g})")
        elif random_forest:
            typer.echo(
                f"Random Forest con criterio Poisson (configuración {forest_config}): "
                f"{FOREST_CONFIGS[forest_config]}"
            )
        else:
            typer.echo(f"Referencia (media de entrenamiento): {predictions[0]:.4f}")
        typer.echo(f"MAE en {period}: {mae:.4f} salidas")
        typer.echo(f"Sesgo medio en {period}: {bias:+.4f} salidas")
        typer.echo(f"Características: {feature_set} ({n_features} columnas)")
        for label in ("known_station", "new_station"):
            if f"{split}_mae_{label}" in detail:
                typer.echo(f"MAE {label}: {detail[f'{split}_mae_{label}']:.4f}")

        if experiment:
            import mlflow

            database = PROJ_ROOT / "data" / "mlflow.db"
            if tracking_uri is None:
                database.parent.mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri(tracking_uri or f"sqlite:///{database.as_posix()}")
            mlflow.set_experiment(experiment)
            with mlflow.start_run(run_name=f"{model_label}-{split}") as run:
                mlflow.log_params(
                    {
                        **model.get_params(deep=False),
                        "model": type(model).__name__,
                        "model_config": model_label,
                        "evaluation_split": split,
                        "input_sha256": sha256(input_path),
                        "train_end": train_end,
                        "validation_end": validation_end,
                        "train_rows": len(train),
                        "evaluation_rows": len(evaluation),
                        "predictors": ",".join(inputs),
                        "feature_set": feature_set,
                        "feature_steps": repr(feature_steps or "CalendarStep (v1)"),
                        "n_features": n_features,
                        "catalog_sha256": sha256(catalog_path) if catalog is not None else "",
                    }
                )
                mlflow.log_metrics(detail)
                if register_model:
                    import mlflow.sklearn

                    model_info = mlflow.sklearn.log_model(
                        pipeline,
                        name="pipeline",
                        input_example=train[inputs].head(3),
                        serialization_format="cloudpickle",
                    )
                    version = mlflow.register_model(
                        model_info.model_uri, register_model
                    )
                    typer.echo(
                        f"Modelo registrado: models:/{version.name}/{version.version}"
                    )
                typer.echo(f"MLflow run_id: {run.info.run_id}")


if __name__ == "__main__":
    app()
