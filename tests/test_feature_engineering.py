from datetime import date

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

from MiBici.feature_engineering import (
    AnnualLagStep,
    CatalogStep,
    HolidayStep,
    StationAgeStep,
    StationProfileStep,
    build_steps,
    easter_date,
    load_station_catalog,
    mexican_holidays,
    run_steps,
)
from MiBici.features import PREDICTORS, build_calendar_features
from MiBici.modeling.train import FEATURE_INPUTS, build_model_pipeline


def history():
    dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    rows = []
    for station, level in ((1, 10), (2, 30), (3, 5)):
        for day in dates:
            rows.append((day, station, level + (day.day_of_week >= 5) * 2))
    return build_calendar_features(
        pd.DataFrame(rows, columns=["fecha", "estacion_id", "salidas"])
    )


def catalog(tmp_path):
    path = tmp_path / "nomenclatura.csv"
    pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7],
        "name": ["(GDL-001) A", "(GDL-002) B", "(ZPN-001) C", "(TLQ-001) D",
                 "(GDL-003) E", "(GDL-004) F", "Sin prefijo"],
        "latitude": [20.67, 20.68, 20.70, 20.64, 20.671, 20.69, 20.66],
        "longitude": [-103.35, -103.36, -103.40, -103.31, -103.351, -103.37, -103.34],
        "dpcapacity": [15, 20, 0, 12, 15, 15, 10],
    }).to_csv(path, index=False)
    return load_station_catalog(path)


def test_known_mexican_holidays():
    holidays = mexican_holidays([2026])
    assert {date(2026, 2, 2), date(2026, 3, 16), date(2026, 11, 16)} <= holidays
    assert easter_date(2026) == date(2026, 4, 5)
    assert easter_date(2025) == date(2025, 4, 20)


def test_holiday_step_flags():
    data = pd.DataFrame({
        "fecha": pd.to_datetime(["2026-03-16", "2026-04-03", "2026-12-24", "2026-06-10"]),
        "estacion_id": ["1"] * 4,
    })
    out = HolidayStep().transform(data)
    assert out.festivo.tolist() == [1, 0, 0, 0]
    assert out.semana_santa.tolist() == [0, 1, 0, 0]
    assert out.vacaciones_invierno.tolist() == [0, 0, 1, 0]


def test_station_age_does_not_extrapolate_year():
    train = history()
    step = StationAgeStep().fit(train)
    future = build_calendar_features(pd.DataFrame({
        "fecha": pd.to_datetime(["2027-05-01", "2027-05-01"]),
        "estacion_id": [1, 99],
        "salidas": [0, 0],
    }))
    out = step.transform(future)
    assert out.anio_cat.tolist() == ["2025", "2025"]
    assert out.estacion_nueva.tolist() == [0, 1]
    assert out.antiguedad_log.iloc[0] > 0 and out.antiguedad_log.iloc[1] == 0


def test_profile_ignores_evaluation_targets():
    train = history()
    step = StationProfileStep().fit(train)
    evaluation = build_calendar_features(pd.DataFrame({
        "fecha": pd.to_datetime(["2026-01-05", "2026-01-05"]),
        "estacion_id": [1, 42],
        "salidas": [1000, 1000],
    }))
    first = step.transform(evaluation)
    second = step.transform(evaluation.assign(salidas=[0, 0]))
    pd.testing.assert_frame_equal(first.drop(columns="salidas"), second.drop(columns="salidas"))
    assert np.isclose(first.perfil_log_media.iloc[1], step.fallback_["perfil_log_media"])


def test_annual_lag_uses_previous_year_or_fallback():
    train = history()
    chained = run_steps(train, [StationProfileStep(), AnnualLagStep()])
    assert {"perfil_log_media", "rezago_anual_log"} <= set(chained.columns)
    profile = StationProfileStep().fit(train)
    lag = AnnualLagStep().fit(train)
    query = build_calendar_features(pd.DataFrame({
        "fecha": pd.to_datetime(["2026-02-10", "2028-02-10", "2026-02-10"]),
        "estacion_id": [2, 2, 77],
        "salidas": [0, 0, 0],
    }))
    out = lag.transform(profile.transform(query))
    february_2025 = train.loc[(train.estacion_id == "2") & (train.fecha.dt.strftime("%Y-%m") == "2025-02")]
    assert np.isclose(out.rezago_anual_log.iloc[0], np.log1p(february_2025.salidas.mean()))
    assert np.isclose(out.rezago_anual_log.iloc[1], out.rezago_anual_log.iloc[0])
    assert out.rezago_disponible.tolist() == [1, 1, 0]
    assert np.isclose(out.rezago_anual_log.iloc[2], out.perfil_log_media.iloc[2])


def test_catalog_gives_new_stations_neighbor_estimate(tmp_path):
    train = history()
    stations = catalog(tmp_path)
    step = CatalogStep(stations, k=2).fit(train)
    query = build_calendar_features(pd.DataFrame({
        "fecha": pd.to_datetime(["2026-01-05"] * 3),
        "estacion_id": [5, 7, 500],
        "salidas": [0, 0, 0],
    }))
    out = step.transform(query)
    assert out.zona.tolist() == ["GDL", "OTRA", "DESCONOCIDA"]
    assert out.vecinos_log_media.notna().all()
    assert stations.loc["3", "dpcapacity"] == 0
    assert step.table_.loc["3", "capacidad"] > 0


def test_v1_steps_reproduce_legacy_pipeline():
    train = history()
    legacy = build_model_pipeline(PoissonRegressor(alpha=0.01, max_iter=500))
    legacy.fit(train[PREDICTORS], train.salidas)
    modular = build_model_pipeline(
        PoissonRegressor(alpha=0.01, max_iter=500), build_steps("v1")
    )
    modular.fit(train[FEATURE_INPUTS], train.salidas)
    np.testing.assert_allclose(
        legacy.predict(train[PREDICTORS]), modular.predict(train[FEATURE_INPUTS])
    )


def test_every_feature_set_trains_and_predicts(tmp_path):
    train = history()
    stations = catalog(tmp_path)
    future = build_calendar_features(pd.DataFrame({
        "fecha": pd.to_datetime(["2026-03-16", "2026-11-20"]),
        "estacion_id": [1, 5],
        "salidas": [0, 0],
    }))
    for name in ("v2_calendario", "v3_tendencia", "v4_perfil_estacion",
                 "v5_interacciones", "v6_catalogo_espacial", "v7_rezago_anual"):
        pipeline = build_model_pipeline(
            PoissonRegressor(alpha=0.01, max_iter=500), build_steps(name, stations)
        )
        pipeline.fit(train[FEATURE_INPUTS], train.salidas)
        predictions = pipeline.predict(future[FEATURE_INPUTS])
        assert predictions.shape == (2,) and np.isfinite(predictions).all(), name
