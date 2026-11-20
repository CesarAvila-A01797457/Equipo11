import pandas as pd
from sklearn.linear_model import PoissonRegressor

from MiBici.features import (
    PREDICTORS,
    build_calendar_features,
    load_daily_counts,
    temporal_split,
)
from MiBici.modeling.train import build_model_pipeline, calculate_metrics


def test_csv_to_predictions_and_metrics(tmp_path):
    # El archivo pequeño permite probar el flujo sin descargar el histórico.
    records = [
        ("2024-01-01", 1, 3),
        ("2024-01-02", 1, 4),
        ("2024-01-03", 1, 2),
        ("2024-01-04", 2, 5),
        ("2024-02-01", 1, 3),
        ("2024-02-02", 2, 4),
        ("2024-02-03", 3, 6),
        ("2024-02-04", 1, 2),
        ("2024-03-01", 1, 4),
        ("2024-03-02", 2, 3),
        ("2024-03-03", 3, 5),
        ("2024-03-04", 1, 6),
    ]
    input_path = tmp_path / "salidas.csv"
    pd.DataFrame(records, columns=["fecha", "estacion_id", "salidas"]).to_csv(
        input_path, index=False
    )

    data = load_daily_counts(input_path)
    frame = build_calendar_features(data)
    splits = temporal_split(frame, "2024-01-31", "2024-02-29")
    pipeline = build_model_pipeline(PoissonRegressor(alpha=0.01, max_iter=500))
    pipeline.fit(splits["train"][PREDICTORS], splits["train"]["salidas"])

    actual = splits["test"]["salidas"].to_numpy()
    predictions = pipeline.predict(splits["test"][PREDICTORS])
    metrics = calculate_metrics(actual, predictions)

    assert len(predictions) == len(splits["test"])
    assert (predictions >= 0).all()
    assert metrics["mae"] >= 0
    assert pd.notna(metrics["bias"])
