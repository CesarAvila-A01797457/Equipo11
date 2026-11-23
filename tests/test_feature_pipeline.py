import numpy as np
import pandas as pd
import pytest

from MiBici.feature_engineering import CalendarStep, FeaturePipeline, FeatureStep
from MiBici.features import PREDICTORS, build_calendar_features


def frame():
    return pd.DataFrame({
        "fecha": pd.to_datetime(["2025-01-06", "2025-01-11", "2025-02-03"]),
        "estacion_id": [1, 2, 1],
        "salidas": [3, 5, 4],
    })


class AddOne(FeatureStep):
    numeric = ("uno",)

    def transform(self, data):
        return data.assign(uno=1.0)


class UsesPrevious(FeatureStep):
    numeric = ("doble",)

    def transform(self, data):
        return data.assign(doble=data["uno"] * 2)


class DropsRow(FeatureStep):
    def transform(self, data):
        return data.iloc[1:]


class RecordsFit(FeatureStep):
    def fit(self, data):
        self.seen_target_ = "salidas" in data
        return self

    def transform(self, data):
        self.transform_saw_target_ = "salidas" in data
        return data


def test_build_calendar_features_matches_v1_definition():
    data = frame()
    result = build_calendar_features(data)
    expected = data.copy()
    expected["estacion_id"] = expected.estacion_id.astype("string")
    expected["dia_semana"] = expected.fecha.dt.day_of_week.astype("string")
    expected["mes"] = expected.fecha.dt.month.astype("string")
    expected["anio"] = expected.fecha.dt.year.astype("int64")
    pd.testing.assert_frame_equal(result, expected)
    assert PREDICTORS == ["estacion_id", "dia_semana", "mes", "anio"]


def test_steps_run_in_order_and_see_previous_columns():
    pipeline = FeaturePipeline([CalendarStep(), AddOne(), UsesPrevious()])
    data = frame()
    out = pipeline.fit(data[["fecha", "estacion_id"]], data.salidas).transform(
        data[["fecha", "estacion_id"]]
    )
    assert (out["doble"] == 2.0).all()
    assert pipeline.numeric_columns() == ["anio", "uno", "doble"]


def test_fit_sees_target_but_transform_does_not():
    step = RecordsFit()
    data = frame()
    pipeline = FeaturePipeline([step]).fit(data[["fecha", "estacion_id"]], data.salidas)
    assert step.seen_target_
    pipeline.transform(data)
    assert not step.transform_saw_target_


def test_rejects_steps_that_change_rows():
    data = frame()
    with pytest.raises(ValueError, match="filas"):
        FeaturePipeline([DropsRow()]).fit(data[["fecha", "estacion_id"]], data.salidas)


def test_rejects_duplicated_columns():
    data = frame()
    with pytest.raises(ValueError, match="dos veces"):
        FeaturePipeline([AddOne(), AddOne()]).fit(data, data.salidas)


def test_column_transformer_uses_declared_columns():
    data = frame()
    pipeline = FeaturePipeline([CalendarStep(), AddOne()])
    inputs = data[["fecha", "estacion_id"]]
    transformed = pipeline.fit(inputs, data.salidas).transform(inputs)
    columns = pipeline.build_column_transformer().fit(transformed)
    names = columns.get_feature_names_out()
    assert any(name.startswith("categoria__estacion_id") for name in names)
    assert {"numero__anio", "numero__uno"} <= set(names)
    assert np.isfinite(columns.transform(transformed).toarray()).all()
