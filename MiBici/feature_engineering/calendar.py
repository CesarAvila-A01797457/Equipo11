"""Variables de calendario básicas del modelo v1."""

from MiBici.feature_engineering.base import FeatureStep


class CalendarStep(FeatureStep):
    """Estación como texto, día de la semana, mes y año a partir de ``fecha``."""

    categorical = ("estacion_id", "dia_semana", "mes")
    numeric = ("anio",)

    def transform(self, frame):
        frame = frame.copy()
        frame["estacion_id"] = frame.estacion_id.astype("string")
        frame["dia_semana"] = frame.fecha.dt.day_of_week.astype("string")
        frame["mes"] = frame.fecha.dt.month.astype("string")
        frame["anio"] = frame.fecha.dt.year.astype("int64")
        return frame
