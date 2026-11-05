"""Prepara los cortes temporales y evalúa modelos para Fase 2."""

from pathlib import Path

import typer
from sklearn.dummy import DummyRegressor
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


@app.command()
def main(
    input_path: Path = FILTERED_DIR / "mibici_salidas_estacion_fecha.csv",
    quality_report: Path = FILTERED_DIR / "mibici_quality_report.json",
    train_end: str = "2025-12-31",
    validation_end: str = "2026-06-30",
    poisson: bool = typer.Option(False, help="Evalúa una regresión de Poisson."),
    alpha: float = typer.Option(0.1, min=0.0, help="Regularización de Poisson."),
    test: bool = typer.Option(False, help="Evalúa el modelo elegido en el periodo de prueba."),
    experiment: str | None = typer.Option(None, help="Experimento de MLflow para esta corrida."),
    tracking_uri: str | None = typer.Option(None, help="URI de seguimiento de MLflow."),
    register_model: str | None = typer.Option(None, help="Nombre en MLflow Model Registry."),
    baseline: bool = typer.Option(False, help="Evalúa la media de entrenamiento en validación."),
):
    """Valida las particiones temporales y evalúa el modelo solicitado."""
    if baseline and poisson:
        raise typer.BadParameter("Elige --baseline o --poisson para cada corrida.")

    if test and not (baseline or poisson):
        raise typer.BadParameter("--test requiere --baseline o --poisson.")

    if experiment and not (baseline or poisson):
        raise typer.BadParameter("--experiment requiere --baseline o --poisson.")

    if register_model and not (poisson and test and experiment):
        raise typer.BadParameter("--register-model requiere --poisson, --test y --experiment.")

    data = load_daily_counts(input_path, quality_report)
    parts = temporal_split(build_calendar_features(data), train_end, validation_end)
    for name, part in parts.items():
        first_date = part.fecha.min().date()
        last_date = part.fecha.max().date()
        typer.echo(f"{name}: {len(part)} filas ({first_date} a {last_date})")

    if baseline or poisson:
        if poisson:
            model = PoissonRegressor(alpha=alpha, max_iter=500)
        else:
            model = DummyRegressor(strategy="mean")
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
            with mlflow.start_run(run_name=f"{type(model).__name__}-{split}") as run:
                mlflow.log_params(
                    {
                        **model.get_params(deep=False),
                        "model": type(model).__name__,
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
                    version = mlflow.register_model(model_info.model_uri, register_model)
                    typer.echo(f"Modelo registrado: models:/{version.name}/{version.version}")
                typer.echo(f"MLflow run_id: {run.info.run_id}")


if __name__ == "__main__":
    app()
