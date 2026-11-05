"""Prepara los cortes temporales y evalúa modelos para Fase 2."""

from pathlib import Path

import typer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

from MiBici.config import PROCESSED_DATA_DIR, PROJ_ROOT
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
                random_state=42,
                n_jobs=-1,
                **FOREST_CONFIGS[forest_config],
            )
            model_label = f"random-forest-config-{forest_config}"
        else:
            model = DummyRegressor(strategy="mean")
            model_label = "baseline-mean"
        train = parts["train"]
        split = "test" if test else "validation"
        evaluation = parts[split]
        period = "prueba" if test else "validación"
        pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                ("model", model),
            ]
        )
        pipeline.fit(train[PREDICTORS], train["salidas"])
        predictions = pipeline.predict(evaluation[PREDICTORS])
        actual = evaluation["salidas"].to_numpy()
        mae = mean_absolute_error(actual, predictions)
        bias = (predictions - actual).mean()
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
                        "predictors": ",".join(PREDICTORS),
                    }
                )
                mlflow.log_metrics({f"{split}_mae": mae, f"{split}_bias": bias})
                if register_model:
                    import mlflow.sklearn

                    model_info = mlflow.sklearn.log_model(
                        pipeline,
                        name="pipeline",
                        input_example=train[PREDICTORS].head(3),
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
