"""Días atípicos del calendario mexicano calculados por regla."""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from MiBici.feature_engineering.base import FeatureStep


def easter_date(year: int) -> date:
    """Domingo de Pascua (algoritmo gregoriano anónimo)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def nth_monday(year: int, month: int, n: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(7 - first.weekday()) % 7 + 7 * (n - 1))


def mexican_holidays(years) -> set[date]:
    """Descansos obligatorios de la Ley Federal del Trabajo (reglas desde 2006)."""
    holidays = set()
    for year in years:
        holidays |= {
            date(year, 1, 1),
            nth_monday(year, 2, 1),
            nth_monday(year, 3, 3),
            date(year, 5, 1),
            date(year, 9, 16),
            nth_monday(year, 11, 3),
            date(year, 12, 25),
        }
        # Transmisión del Poder Ejecutivo Federal.
        if year == 2018:
            holidays.add(date(2018, 12, 1))
        if year >= 2024 and (year - 2024) % 6 == 0:
            holidays.add(date(year, 10, 1))
    return holidays


class HolidayStep(FeatureStep):
    """Festivo, Semana Santa, vacaciones de invierno y días de quincena."""

    numeric = ("festivo", "semana_santa", "vacaciones_invierno", "quincena")

    def transform(self, frame):
        frame = frame.copy()
        fecha = frame.fecha.dt.normalize()
        years = range(fecha.dt.year.min() - 1, fecha.dt.year.max() + 2)
        holidays = pd.to_datetime(sorted(mexican_holidays(years)))
        easter = pd.to_datetime([easter_date(year) for year in years])
        # Domingo de Ramos a Domingo de Pascua, y la semana siguiente (Pascua).
        holy = pd.DatetimeIndex(np.concatenate([
            (easter + pd.Timedelta(days=offset)).to_numpy()
            for offset in range(-7, 8)
        ]))
        day, month = fecha.dt.day, fecha.dt.month
        frame["festivo"] = fecha.isin(holidays).astype("int8")
        frame["semana_santa"] = fecha.isin(holy).astype("int8")
        frame["vacaciones_invierno"] = (
            ((month == 12) & (day >= 20)) | ((month == 1) & (day <= 6))
        ).astype("int8")
        frame["quincena"] = (
            day.isin([1, 2, 15, 16]) | (day == fecha.dt.days_in_month)
        ).astype("int8")
        frame["fin_de_semana"] = (fecha.dt.day_of_week >= 5).astype("int8").astype("string")
        return frame
