"""Variables del catálogo de estaciones y vecindario espacial."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from MiBici.feature_engineering.base import FeatureStep

EARTH_RADIUS_KM = 6371.0


def load_station_catalog(path) -> pd.DataFrame:
    """Lee la nomenclatura y deriva la zona del prefijo del nombre (GDL, ZPN...)."""
    catalog = pd.read_csv(
        Path(path), usecols=["id", "name", "latitude", "longitude", "dpcapacity"]
    )
    if catalog.id.duplicated().any() or catalog.isna().any().any():
        raise ValueError("El catálogo tiene IDs repetidos o campos vacíos")
    catalog["estacion_id"] = catalog.id.astype("int64").astype("string")
    catalog["zona"] = (
        catalog.name.str.extract(r"^\(([A-Z]+)-", expand=False).fillna("OTRA")
    )
    return catalog.set_index("estacion_id")[
        ["zona", "latitude", "longitude", "dpcapacity"]
    ]


class CatalogStep(FeatureStep):
    """Capacidad, zona, distancia al centro y demanda de las k estaciones vecinas.

    La demanda vecina usa la media reciente de entrenamiento de las ``k``
    estaciones conocidas más cercanas (sin incluir a la propia), por lo que
    también da una estimación a estaciones nuevas que sí están en el catálogo.
    """

    categorical = ("zona",)
    numeric = ("capacidad", "dist_centro_km", "vecinos_log_media")

    def __init__(self, catalog: pd.DataFrame, k: int = 5, window_days: int = 365):
        self.catalog = catalog
        self.k = k
        self.window_days = window_days

    def fit(self, frame):
        catalog = self.catalog
        end = frame.fecha.max()
        recent = frame.loc[frame.fecha > end - np.timedelta64(self.window_days, "D")]
        level = np.log1p(recent.groupby("estacion_id").salidas.mean())
        known = catalog.loc[catalog.index.isin(level.index)]
        if len(known) < 2:
            raise ValueError("Se necesitan al menos dos estaciones conocidas en el catálogo")
        # Con pocas estaciones conocidas se usan todas las demás como vecinas.
        k = min(self.k, len(known) - 1)
        radians = np.radians(catalog[["latitude", "longitude"]].to_numpy())
        tree = BallTree(np.radians(known[["latitude", "longitude"]].to_numpy()),
                        metric="haversine")
        _, index = tree.query(radians, k=k + 1)
        neighbor_ids = known.index.to_numpy()[index]
        neighbor_level = level.reindex(neighbor_ids.ravel()).to_numpy().reshape(
            neighbor_ids.shape
        )
        own = neighbor_ids == catalog.index.to_numpy()[:, None]
        neighbor_level = np.where(own, np.nan, neighbor_level)
        # Conserva las k más cercanas que no sean la propia estación.
        values = np.array([row[~np.isnan(row)][:k].mean() for row in neighbor_level])
        center = np.radians([catalog.latitude.median(), catalog.longitude.median()])
        distance = BallTree(center[None, :], metric="haversine").query(radians)[0][:, 0]
        capacity = catalog.dpcapacity.where(catalog.dpcapacity > 0)
        self.table_ = pd.DataFrame(
            {
                "zona": catalog.zona,
                "capacidad": capacity.fillna(capacity.median()),
                "dist_centro_km": distance * EARTH_RADIUS_KM,
                "vecinos_log_media": values,
            },
            index=catalog.index,
        )
        self.fallback_ = self.table_[list(self.numeric)].median()
        return self

    def transform(self, frame):
        frame = frame.copy()
        station = frame.estacion_id.astype("string")
        frame["zona"] = station.map(self.table_.zona).fillna("DESCONOCIDA").astype("string")
        for column in self.numeric:
            frame[column] = (
                station.map(self.table_[column]).astype(float).fillna(self.fallback_[column])
            )
        return frame

    def __repr__(self):
        return f"CatalogStep(k={self.k}, estaciones={len(self.catalog)})"
