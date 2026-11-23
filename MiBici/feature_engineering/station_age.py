"""Antigüedad de la estación y año como categoría sin extrapolar."""

import numpy as np

from MiBici.feature_engineering.base import FeatureStep


class StationAgeStep(FeatureStep):
    """Sustituye el año lineal por antigüedad de la estación y año categórico.

    Los años posteriores al último de entrenamiento se asignan a ese último año
    para no extrapolar una tendencia exponencial con el enlace log de Poisson.
    """

    categorical = ("anio_cat",)
    numeric = ("antiguedad_log", "estacion_nueva")
    drops = ("anio",)

    def fit(self, frame):
        self.first_date_ = frame.groupby("estacion_id").fecha.min()
        self.last_year_ = int(frame.fecha.dt.year.max())
        return self

    def transform(self, frame):
        frame = frame.copy()
        first = frame.estacion_id.map(self.first_date_)
        days = (frame.fecha - first).dt.days.clip(lower=0)
        frame["estacion_nueva"] = first.isna().astype("int8")
        frame["antiguedad_log"] = np.log1p(days.fillna(0).astype(float))
        frame["anio_cat"] = (
            frame.fecha.dt.year.clip(upper=self.last_year_).astype("string")
        )
        return frame
