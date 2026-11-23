"""Interfaz común de los pasos de ingeniería de características."""

import pandas as pd


class FeatureStep:
    """Recibe el conjunto de datos, lo modifica y lo devuelve al siguiente paso.

    Cada paso declara las columnas que aporta como predictores:
    ``categorical`` se codifica con one-hot, ``numeric`` se escala y ``drops``
    retira columnas declaradas por pasos anteriores.

    ``fit`` recibe el conjunto de entrenamiento con la columna ``salidas`` y solo
    debe guardar tablas de consulta. ``transform`` no debe leer ``salidas``: se
    aplica igual en validación, prueba y en la API, donde no hay objetivo.
    """

    categorical: tuple[str, ...] = ()
    numeric: tuple[str, ...] = ()
    drops: tuple[str, ...] = ()

    def fit(self, frame: pd.DataFrame) -> "FeatureStep":
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
