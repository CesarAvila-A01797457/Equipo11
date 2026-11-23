"""Salidas de la misma estación en el mismo mes de años anteriores."""

import numpy as np
import pandas as pd

from MiBici.feature_engineering.base import FeatureStep


class AnnualLagStep(FeatureStep):
    """Log de la media del mismo mes en el año más reciente anterior disponible.

    Solo consulta meses de entrenamiento. Sin historial usa ``perfil_log_media``
    si un paso anterior la creó; si no, la mediana de entrenamiento.
    """

    numeric = ("rezago_anual_log", "rezago_disponible")

    def fit(self, frame):
        monthly = (
            frame.assign(anio_ref=frame.fecha.dt.year, mes_ref=frame.fecha.dt.month)
            .groupby(["estacion_id", "mes_ref", "anio_ref"])
            .salidas.mean()
        )
        self.table_ = (
            np.log1p(monthly).rename("rezago_anual_log").reset_index()
            .astype({"estacion_id": "string", "anio_ref": "int64", "mes_ref": "int64"})
            .sort_values("anio_ref")
        )
        self.fallback_ = float(self.table_.rezago_anual_log.median())
        return self

    def transform(self, frame):
        frame = frame.copy()
        query = pd.DataFrame({
            "row": np.arange(len(frame)),
            "estacion_id": frame.estacion_id.astype("string").to_numpy(),
            "mes_ref": frame.fecha.dt.month.astype("int64").to_numpy(),
            "anio_ref": (frame.fecha.dt.year - 1).astype("int64").to_numpy(),
        }).astype({"estacion_id": "string"}).sort_values("anio_ref")
        matched = pd.merge_asof(
            query, self.table_, on="anio_ref", by=["estacion_id", "mes_ref"],
            direction="backward",
        ).sort_values("row")
        lag = pd.Series(matched.rezago_anual_log.to_numpy(), index=frame.index)
        fallback = (
            frame["perfil_log_media"] if "perfil_log_media" in frame
            else pd.Series(self.fallback_, index=frame.index)
        )
        frame["rezago_disponible"] = lag.notna().astype("int8")
        frame["rezago_anual_log"] = lag.fillna(fallback).astype(float)
        return frame
