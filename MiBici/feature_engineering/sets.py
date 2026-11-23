"""Conjuntos de características acumulativos para los experimentos."""

from MiBici.feature_engineering.annual_lag import AnnualLagStep
from MiBici.feature_engineering.calendar import CalendarStep
from MiBici.feature_engineering.catalog import CatalogStep
from MiBici.feature_engineering.holidays import HolidayStep
from MiBici.feature_engineering.interactions import InteractionStep
from MiBici.feature_engineering.station_age import StationAgeStep
from MiBici.feature_engineering.station_profile import StationProfileStep


def _v2(catalog):
    return [CalendarStep(), HolidayStep()]


def _v3(catalog):
    return _v2(catalog) + [StationAgeStep()]


def _v4(catalog):
    return _v3(catalog) + [StationProfileStep()]


def _v5(catalog):
    return _v4(catalog) + [InteractionStep([("estacion_id", "fin_de_semana")])]


def _v6(catalog):
    if catalog is None:
        raise ValueError("v6_catalogo_espacial requiere el catálogo de estaciones")
    return _v5(catalog) + [
        CatalogStep(catalog),
        InteractionStep([("zona", "dia_semana")]),
    ]


def _v7(catalog):
    return _v6(catalog) + [AnnualLagStep()]


FEATURE_SETS = {
    "v1": lambda catalog: [CalendarStep()],
    "v2_calendario": _v2,
    "v3_tendencia": _v3,
    "v4_perfil_estacion": _v4,
    "v5_interacciones": _v5,
    "v6_catalogo_espacial": _v6,
    "v7_rezago_anual": _v7,
}
NEEDS_CATALOG = {"v6_catalogo_espacial", "v7_rezago_anual"}


def build_steps(feature_set: str, catalog=None):
    """Devuelve instancias nuevas de los pasos de ``feature_set``."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Conjunto desconocido: {feature_set}")
    return FEATURE_SETS[feature_set](catalog)
