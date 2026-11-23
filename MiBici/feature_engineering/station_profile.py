"""Perfil reciente de cada estación aprendido en entrenamiento."""

import numpy as np

from MiBici.feature_engineering.base import FeatureStep


class StationProfileStep(FeatureStep):
    """Nivel, peso de fin de semana y actividad del último año de entrenamiento."""

    numeric = ("perfil_log_media", "perfil_fin_semana", "perfil_dias_activos")

    def __init__(self, window_days: int = 365):
        self.window_days = window_days

    def fit(self, frame):
        end = frame.fecha.max()
        recent = frame.loc[frame.fecha > end - np.timedelta64(self.window_days, "D")]
        weekend = recent.fecha.dt.day_of_week >= 5
        grouped = recent.assign(
            salidas_fs=recent.salidas.where(weekend, 0)
        ).groupby("estacion_id")
        span = (end - grouped.fecha.min()).dt.days + 1
        profile = grouped.agg(
            media=("salidas", "mean"),
            total=("salidas", "sum"),
            total_fs=("salidas_fs", "sum"),
            dias=("fecha", "nunique"),
        )
        self.profile_ = profile.assign(
            perfil_log_media=np.log1p(profile.media),
            perfil_fin_semana=profile.total_fs / profile.total.where(profile.total > 0),
            perfil_dias_activos=profile.dias / span,
        )[list(self.numeric)].fillna(0.0)
        self.fallback_ = self.profile_.median()
        return self

    def transform(self, frame):
        frame = frame.copy()
        for column in self.numeric:
            frame[column] = (
                frame.estacion_id.map(self.profile_[column])
                .astype(float)
                .fillna(self.fallback_[column])
            )
        return frame
