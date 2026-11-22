import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import PoissonRegressor

from MiBici.drift import categorical_distance, corrupt_station_ids, evaluate_scenarios
from MiBici.features import PREDICTORS, build_calendar_features
from MiBici.modeling.train import build_model_pipeline


def sample_frame():
    return build_calendar_features(pd.DataFrame({
        "fecha": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"]),
        "estacion_id": [1, 2, 1, 2],
        "salidas": [5, 40, 7, 45],
    }))


def test_corruption_preserves_targets_dates_keys_and_original():
    frame = sample_frame()
    original = frame.copy(deep=True)
    altered, count = corrupt_station_ids(frame, {"1", "2"}, 1.0, 42)
    assert count == 4
    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_frame_equal(
        altered.drop(columns="estacion_id"), original.drop(columns="estacion_id")
    )
    assert not altered.estacion_id.isin({"1", "2"}).any()
    assert not altered.duplicated(["fecha", "estacion_id"]).any()
    assert categorical_distance(frame.estacion_id, altered.estacion_id) == pytest.approx(1)


def test_corruption_is_reproducible_and_does_not_select_unknown_rows():
    frame = sample_frame()
    frame.loc[0, "estacion_id"] = "99"
    first, count = corrupt_station_ids(frame, {"1", "2"}, 0.7, 42)
    second, _ = corrupt_station_ids(frame, {"1", "2"}, 0.7, 42)
    pd.testing.assert_frame_equal(first, second)
    assert count == 2
    assert first.loc[0, "estacion_id"] == "99"
    with pytest.raises(ValueError):
        corrupt_station_ids(frame, {"1", "2"}, 1.1, 42)


def test_frozen_pipeline_metrics_and_alerts():
    frame = sample_frame()
    model = build_model_pipeline(PoissonRegressor(alpha=0.01, max_iter=500))
    model.fit(frame[PREDICTORS], frame.salidas)
    original_predictions = model.predict(frame[PREDICTORS])
    rows = evaluate_scenarios(model, frame, [1.0], 42, 0.1, 0.1)
    np.testing.assert_array_equal(original_predictions, model.predict(frame[PREDICTORS]))
    assert rows[0]["changed_rows"] == 0
    assert not rows[0]["joint_alert"]
    assert rows[1]["drift_alert"]
    assert rows[1]["performance_alert"]
    assert rows[1]["joint_alert"]
    assert rows[1]["mae"] > rows[0]["mae"]
